import os
import sqlite3
import datetime
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, g
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.secret_key = 'dev_key_123'
DATABASE = os.getenv("DATABASE", "database.db")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o-mini"
# GEMINI_MODEL_NAME = "gemini-pro"
# GEMINI_MODEL_NAME = "gemini-1.5-flash"
GEMINI_MODEL_NAME = 'gemini-2.5-flash'

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✓ OpenAI initialized")
except Exception as e:
    openai_client = None
    print(f"ERROR OpenAI: {e}")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai
        print("✓ Gemini initialized")
    else:
        gemini_client = None
except Exception as e:
    gemini_client = None


def get_ai_provider():
    provider = request.form.get('ai_provider') or session.get('ai_provider', 'openai')
    if provider not in ['openai', 'gemini']:
        provider = 'openai'
    session['ai_provider'] = provider
    return provider


def ai_generic_chat_reply(user_message, ai_provider=None):
    provider = ai_provider or get_ai_provider()
    if provider == 'gemini':
        if not gemini_client:
            return "⚠️ Gemini API Key not configured."
        try:
            model = gemini_client.GenerativeModel(GEMINI_MODEL_NAME)
            return model.generate_content(user_message).text
        except Exception as e:
            return f"Error (Gemini): {str(e)}"
    else:
        if not openai_client:
            return "⚠️ OpenAI API Key not configured."
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Respond in plain text only. Do not use Markdown formatting."},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error (OpenAI): {str(e)}"


def ai_pdf_chat_reply(user_query, notes_text="", pdf_text="", ai_provider=None):
    provider = ai_provider or get_ai_provider()
    context = pdf_text if pdf_text and len(pdf_text.strip()) > 10 else notes_text or "No context provided"
    prompt = f"Context/Notes:\n{context}\n\nUser Question: {user_query}\n\nAnswer based on the context. If not found, say so gracefully."
    if provider == 'gemini':
        if not gemini_client:
            return "⚠️ Gemini API Key not configured."
        try:
            model = gemini_client.GenerativeModel(GEMINI_MODEL_NAME)
            return model.generate_content(prompt).text
        except Exception as e:
            return f"Error (Gemini): {str(e)}"
    else:
        if not openai_client:
            return "⚠️ OpenAI API Key not configured."
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error (OpenAI): {str(e)}"


def generate_quiz(topic, num_questions=5, ai_provider=None):
    provider = ai_provider or 'openai'
    prompt = f"""You are a quiz generator. Create exactly {num_questions} multiple choice questions about the topic: "{topic}"

IMPORTANT: Return ONLY a valid JSON object. No explanation, no markdown, no code blocks. Just raw JSON.

Required format:
{{
  "topic": "{topic}",
  "questions": [
    {{
      "question": "Clear question text ending with ?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer_index": 0
    }}
  ]
}}

Rules:
- Each question must have exactly 4 options
- answer_index is 0-based (0=first option, 1=second, etc.)
- Questions must be factual and clear
- Do NOT include any text before or after the JSON"""
    raw = ""
    if provider == 'gemini' and gemini_client:
        try:
            model = gemini_client.GenerativeModel(GEMINI_MODEL_NAME)
            raw = model.generate_content(prompt).text
        except Exception as e:
            return None, f"Gemini error: {str(e)}"
    elif openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.choices[0].message.content
        except Exception as e:
            return None, f"OpenAI error: {str(e)}"
    else:
        return None, "No AI provider available."
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean), None
    except Exception as e:
        return None, f"Failed to parse quiz: {str(e)}"


def generate_quiz_feedback(topic, score, total):
    pct = score / total if total > 0 else 0
    prompt = f"""A student scored {score}/{total} ({round(pct*100)}%) on a quiz about '{topic}'.
Return ONLY valid JSON in this format, no other text:
{{
  "feedback_text": "2-3 sentence encouraging feedback here",
  "next_steps": ["Step 1", "Step 2", "Step 3"]
}}"""
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.choices[0].message.content
            clean = raw.strip().replace("```json","").replace("```","").strip()
            return json.loads(clean)
        except Exception:
            pass
    # Fallback dict
    return {
        "feedback_text": f"You scored {score} out of {total} on {topic}. {'Great job!' if pct >= 0.7 else 'Keep practicing to improve!'}",
        "next_steps": ["Review incorrect answers", "Study the topic further", "Try another quiz"]
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath):
    try:
        filename = filepath.lower()
        if filename.endswith('.pdf'):
            try:
                import PyPDF2
                text = ""
                with open(filepath, 'rb') as file:
                    for page in PyPDF2.PdfReader(file).pages:
                        try:
                            t = page.extract_text()
                            if t:
                                text += t + "\n"
                        except:
                            pass
                return text if len(text.strip()) > 10 else "[PDF has no extractable text.]"
            except ImportError:
                return "[PyPDF2 not installed.]"
            except Exception as e:
                return f"[PDF error: {str(e)}]"
        elif filename.endswith('.docx'):
            try:
                from docx import Document
                return "\n".join([p.text for p in Document(filepath).paragraphs])
            except ImportError:
                return "[python-docx not installed.]"
        elif filename.endswith(('.txt', '.doc')):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        return "[Unsupported file type]"
    except Exception as e:
        return f"[Error: {str(e)}]"


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );''')
        db.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            session_id TEXT,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );''')
        try:
            db.execute('ALTER TABLE chat_logs ADD COLUMN session_id TEXT')
        except:
            pass
        db.execute('''CREATE TABLE IF NOT EXISTS quiz_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );''')
        db.commit()


@app.before_request
def load_logged_in_user():
    g.history_list = []
    g.current_session_id = None
    user_id = session.get('user_id')
    g.user = None
    if user_id:
        g.user = get_db().execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()


def login_required(view):
    from functools import wraps
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view


def get_chat_history_list(user_id):
    db = get_db()
    rows = db.execute('''
        SELECT session_id, message, created_at
        FROM chat_logs
        WHERE user_id = ? AND mode = ? AND session_id IS NOT NULL
        GROUP BY session_id
        ORDER BY MIN(created_at) DESC
        LIMIT 30
    ''', (user_id, 'generic_chat')).fetchall()
    return [{'session_id': r['session_id'], 'first_message': r['message'], 'created_at': r['created_at']} for r in rows]

def get_pdf_sessions(user_id):
    """Get list of distinct PDF chat sessions."""
    db = get_db()
    rows = db.execute("""
        SELECT session_id, message, response, created_at,
               MIN(created_at) as first_time
        FROM chat_logs
        WHERE user_id = ? AND mode = 'pdf_chat' AND session_id IS NOT NULL
        GROUP BY session_id
        ORDER BY first_time DESC
        LIMIT 20
    """, (user_id,)).fetchall()
    result = []
    for r in rows:
        # Get filename stored in session metadata if available
        meta = db.execute(
            "SELECT response FROM chat_logs WHERE user_id=? AND mode='pdf_meta' AND session_id=? LIMIT 1",
            (user_id, r['session_id'])
        ).fetchone()
        filename = meta['response'] if meta else 'Document'
        result.append({
            'session_id': r['session_id'],
            'first_message': r['message'],
            'filename': filename,
            'created_at': r['created_at']
        })
    return result


@app.route('/new-pdf-chat')
@login_required
def new_pdf_chat():
    """Start a fresh PDF chat session."""
    new_sid = str(uuid.uuid4())
    session['pdf_session_id'] = new_sid
    session.pop('pdf_text', None)
    return redirect(url_for('pdf_chat', session_id=new_sid))





@app.context_processor
def inject_history():
    """Make history_list available in ALL templates automatically."""
    history_list = []
    current_session_id = None
    if hasattr(g, 'user') and g.user:
        try:
            history_list = get_chat_history_list(g.user['id'])
        except Exception:
            history_list = []
    return dict(history_list=history_list)

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db = get_db()
        try:
            db.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', (email, password))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = 'Email already registered.'
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = get_db().execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user is None or user['password_hash'] != password:
            error = 'Invalid email or password.'
        else:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('chat'))
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/new-chat')
@login_required
def new_chat():
    new_sid = str(uuid.uuid4())
    session['chat_session_id'] = new_sid
    return redirect(url_for('chat', session_id=new_sid))


@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    db = get_db()
    session_id = request.args.get('session_id') or session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
    session['chat_session_id'] = session_id

    if request.method == 'POST':
        user_message = request.form.get('message', '').strip()
        if user_message:
            ai_response = ai_generic_chat_reply(user_message)
            db.execute(
                'INSERT INTO chat_logs (user_id, mode, session_id, message, response) VALUES (?, ?, ?, ?, ?)',
                (g.user['id'], 'generic_chat', session_id, user_message, ai_response)
            )
            db.commit()

    logs = db.execute(
        'SELECT * FROM chat_logs WHERE user_id = ? AND mode = ? AND session_id = ? ORDER BY created_at ASC',
        (g.user['id'], 'generic_chat', session_id)
    ).fetchall()

    history_list = get_chat_history_list(g.user['id'])
    return render_template('chat.html', logs=logs, history_list=history_list, current_session_id=session_id)


@app.route('/quiz', methods=['GET', 'POST'])
@login_required
def quiz():
    """Quiz home page - shows stats and form to generate quiz."""
    db = get_db()
    error = None
    if request.method == 'POST':
        topic = request.form.get('topic', '').strip()
        num_questions = int(request.form.get('num_questions', 5))
        if not topic:
            error = "Please enter a topic."
        else:
            quiz_data, err = generate_quiz(topic, num_questions)
            if err or not quiz_data:
                error = err or "Failed to generate quiz."
            else:
                session['quiz'] = quiz_data
                return redirect(url_for('quiz_take'))

    # Stats for sidebar
    total_quizzes = db.execute('SELECT COUNT(*) FROM quiz_sessions WHERE user_id=?', (g.user['id'],)).fetchone()[0]
    avg_row = db.execute('SELECT AVG(CAST(score AS FLOAT)/total_questions) FROM quiz_sessions WHERE user_id=?', (g.user['id'],)).fetchone()
    avg_score = round(avg_row[0] * 100, 1) if avg_row[0] else 0
    history = db.execute('SELECT * FROM quiz_sessions WHERE user_id=? ORDER BY created_at DESC LIMIT 10', (g.user['id'],)).fetchall()
    active_quiz = session.get('quiz')
    return render_template('quiz_home.html', error=error, total_quizzes=total_quizzes,
                           avg_score=avg_score, history=history, active_quiz=active_quiz)


@app.route('/quiz/generate', methods=['POST'])
@login_required
def quiz_generate():
    """Generate quiz from topic - used by quiz_home form."""
    db = get_db()

    def render_home_with_error(error_msg):
        total_quizzes = db.execute('SELECT COUNT(*) FROM quiz_sessions WHERE user_id=?', (g.user['id'],)).fetchone()[0]
        avg_row = db.execute('SELECT AVG(CAST(score AS FLOAT)/total_questions) FROM quiz_sessions WHERE user_id=?', (g.user['id'],)).fetchone()
        avg_score = round(avg_row[0] * 100, 1) if avg_row[0] else 0
        history = db.execute('SELECT * FROM quiz_sessions WHERE user_id=? ORDER BY created_at DESC LIMIT 10', (g.user['id'],)).fetchall()
        return render_template('quiz_home.html', error=error_msg, total_quizzes=total_quizzes,
                               avg_score=avg_score, history=history, active_quiz=session.get('quiz'))

    topic = request.form.get('topic', '').strip()
    if not topic:
        return render_home_with_error("Please enter a topic to generate a quiz.")

    # Always use 5 questions - fixed
    quiz_data, err = generate_quiz(topic, 5)

    if err or not quiz_data:
        return render_home_with_error(f"Could not generate quiz. Please try a clearer topic name.")

    # Validate structure
    if 'questions' not in quiz_data or len(quiz_data['questions']) == 0:
        return render_home_with_error("Quiz generation failed - no questions returned. Please try again.")

    session['quiz'] = quiz_data
    return redirect(url_for('quiz_take'))


@app.route('/quiz/take')
@login_required
def quiz_take():
    quiz_data = session.get('quiz')
    if not quiz_data:
        return redirect(url_for('quiz'))
    return render_template('quiz_take.html', quiz=quiz_data)


@app.route('/quiz/submit', methods=['POST'])
@login_required
def quiz_submit():
    quiz_data = session.get('quiz')
    if not quiz_data:
        return redirect(url_for('quiz'))
    score = 0
    total = len(quiz_data['questions'])
    results = []
    for i, q in enumerate(quiz_data['questions']):
        selected = request.form.get(f'q_{i}')
        user_idx = -1
        if selected and selected.strip():
            try:
                user_idx = int(selected)
            except:
                user_idx = -1
        is_correct = (user_idx == q['answer_index'])
        if is_correct:
            score += 1
        results.append({
            "question": q['question'],
            "selected": q['options'][user_idx] if 0 <= user_idx < len(q['options']) else "Skipped",
            "correct": q['options'][q['answer_index']],
            "is_correct": is_correct
        })
    db = get_db()
    db.execute('INSERT INTO quiz_sessions (user_id, topic, score, total_questions) VALUES (?, ?, ?, ?)',
               (g.user['id'], quiz_data['topic'], score, total))
    db.commit()
    feedback = generate_quiz_feedback(quiz_data['topic'], score, total)
    return render_template('quiz_result.html', score=score, total=total, results=results, feedback=feedback)


@app.route('/pdf-chat', methods=['GET', 'POST'])
@login_required
def pdf_chat():
    db = get_db()
    upload_status = None
    notes_text = ""

    # Get or create session_id for this PDF chat
    pdf_session_id = request.args.get('session_id') or session.get('pdf_session_id')
    if not pdf_session_id:
        pdf_session_id = str(uuid.uuid4())
    session['pdf_session_id'] = pdf_session_id

    # Load pdf_text for this session from session store (keyed by session_id)
    pdf_store_key = f'pdf_text_{pdf_session_id}'
    pdf_text = session.get(pdf_store_key, "")

    if request.method == 'POST':
        user_message = request.form.get('message')
        notes_text = request.form.get('notes_text', "")

        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if file and file.filename != '' and allowed_file(file.filename):
                try:
                    orig_name = secure_filename(file.filename)
                    saved_name = f"{g.user['id']}_{datetime.datetime.now().timestamp()}_{orig_name}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
                    file.save(filepath)
                    pdf_text = extract_text_from_file(filepath)
                    if not pdf_text or len(pdf_text.strip()) < 10:
                        pdf_text = "[No extractable text found.]"
                        upload_status = "⚠️ No text extracted."
                    else:
                        upload_status = f"✓ Uploaded! ({len(pdf_text)} chars extracted)"
                    session[pdf_store_key] = pdf_text
                    # Store filename as metadata
                    db.execute('INSERT INTO chat_logs (user_id, mode, session_id, message, response) VALUES (?, ?, ?, ?, ?)',
                               (g.user['id'], 'pdf_meta', pdf_session_id, 'filename', orig_name))
                    db.commit()
                except Exception as e:
                    upload_status = f"✗ Error: {str(e)}"

        if request.form.get('clear_pdf'):
            session.pop(pdf_store_key, None)
            pdf_text = ""
            upload_status = "PDF cleared!"

        if user_message:
            pdf_text = session.get(pdf_store_key, "")
            ai_response = ai_pdf_chat_reply(user_message, notes_text=notes_text, pdf_text=pdf_text)
            db.execute('INSERT INTO chat_logs (user_id, mode, session_id, message, response) VALUES (?, ?, ?, ?, ?)',
                       (g.user['id'], 'pdf_chat', pdf_session_id, user_message, ai_response))
            db.commit()

    logs = db.execute(
        'SELECT * FROM chat_logs WHERE user_id = ? AND mode = ? AND session_id = ? ORDER BY created_at ASC',
        (g.user['id'], 'pdf_chat', pdf_session_id)
    ).fetchall()

    pdf_text = session.get(pdf_store_key, "")
    pdf_loaded = bool(pdf_text and len(pdf_text) > 10 and "[No extractable" not in pdf_text)
    pdf_sessions = get_pdf_sessions(g.user['id'])

    return render_template('pdf_chat.html',
                           logs=logs,
                           upload_status=upload_status,
                           pdf_loaded=pdf_loaded,
                           pdf_text_preview=pdf_text[:120] if pdf_text else "",
                           notes_text=notes_text,
                           pdf_sessions=pdf_sessions,
                           current_pdf_session=pdf_session_id)


@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    total_quizzes = db.execute('SELECT COUNT(*) FROM quiz_sessions WHERE user_id = ?', (g.user['id'],)).fetchone()[0]
    avg_row = db.execute('SELECT AVG(CAST(score AS FLOAT)/total_questions) FROM quiz_sessions WHERE user_id = ?', (g.user['id'],)).fetchone()
    avg_score = round(avg_row[0] * 100, 1) if avg_row[0] is not None else 0
    recent_quizzes = db.execute('SELECT * FROM quiz_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5', (g.user['id'],)).fetchall()
    return render_template('dashboard.html', total_quizzes=total_quizzes, avg_score=avg_score, recent_quizzes=recent_quizzes)



@app.route('/chat/delete/<session_id>', methods=['POST'])
@login_required
def delete_chat(session_id):
    db = get_db()
    db.execute('DELETE FROM chat_logs WHERE user_id=? AND session_id=? AND mode=?',
               (g.user['id'], session_id, 'generic_chat'))
    db.commit()
    # If deleted current session, clear it
    if session.get('chat_session_id') == session_id:
        session.pop('chat_session_id', None)
    return redirect(url_for('chat'))


@app.route('/pdf-chat/delete/<session_id>', methods=['POST'])
@login_required
def delete_pdf_chat(session_id):
    db = get_db()
    db.execute('DELETE FROM chat_logs WHERE user_id=? AND session_id=?',
               (g.user['id'], session_id))
    db.commit()
    # Clear pdf text for this session
    session.pop(f'pdf_text_{session_id}', None)
    if session.get('pdf_session_id') == session_id:
        session.pop('pdf_session_id', None)
    return redirect(url_for('pdf_chat'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
# NOTE: Already complete above - no changes needed