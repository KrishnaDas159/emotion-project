import sqlite3
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
import re
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import io
# recommendation related libraries and imports
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from api import recommend_music, emotion_dict  # Import your function and mapping

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')

DATABASE = 'users.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

init_db()

ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@example.com')
ADMIN_PASSWORD_HASH = generate_password_hash(os.getenv('ADMIN_PASSWORD', 'adminpassword'))

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def validate_password(password):
    return len(password) >= 8

# Load emotion detection model
model_path = '/home/abhijeet/Documents/emotions/emotion detection/MODEL/model.h5'
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")
else:
    print(f"Error: Model file not found at {model_path}")
    exit()

# Emotion detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_size = 224
emotion_mapping = {0: 'sad', 1: 'fear', 2: 'surprise', 3: 'neutral', 4: 'disgust', 5: 'happy', 6: 'angry'}

# def predict_emotion_from_image(img):
#     """Process an image and return it with emotion annotations."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (img_size, img_size))
#         rgb_face = np.stack((face,) * 3, axis=-1)  # Convert to RGB
#         rgb_face = rgb_face / 255.0
#         rgb_face = np.expand_dims(rgb_face, axis=0)  # Add batch dimension
#         prediction = model.predict(rgb_face, verbose=0)
#         emotion_idx = np.argmax(prediction)
#         emotion_label = emotion_mapping[emotion_idx]
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(img, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#     return img

def predict_emotion_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected_emotions = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (img_size, img_size))
        rgb_face = np.stack((face,) * 3, axis=-1)  # Convert to RGB
        rgb_face = rgb_face / 255.0
        rgb_face = np.expand_dims(rgb_face, axis=0)  # Add batch dimension
        prediction = model.predict(rgb_face, verbose=0)
        emotion_idx = int(np.argmax(prediction))
        detected_emotions.append(emotion_idx)
        emotion_label = emotion_mapping[emotion_idx]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    predicted_emotion = detected_emotions[0] if detected_emotions else None
    return img, predicted_emotion

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if not validate_email(email):
            flash('Invalid email format.', 'danger')
            return redirect(url_for('signup'))
        if not validate_password(password):
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (name, email, hashed_password))
            conn.commit()
            flash('Account created successfully. Please login.', 'success')
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'danger')
        finally:
            conn.close()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash('Login successful!', 'success')
            return redirect(url_for('welcome'))
        else:
            flash('Incorrect email or password.', 'danger')
    return render_template('login.html')

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    return render_template('welcome.html', username=session['user_name'])

# @app.route('/welcome_new')
# def welcome_new():
#     if 'user_id' not in session:
#         flash('Please log in first.', 'warning')
#         return redirect(url_for('login'))
#     return render_template('welcome_UI.html', username=session['user_name'])

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == ADMIN_EMAIL and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['admin'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.', 'danger')
    return render_template('admin.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin'):
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return render_template('admin_dashboard.html', users=users)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()

    if request.method == 'POST':
        new_name = request.form['name']
        new_password = request.form.get('password')

        if new_password:
            hashed_password = generate_password_hash(new_password)
            conn.execute('UPDATE users SET name = ?, password = ? WHERE id = ?', 
                         (new_name, hashed_password, session['user_id']))
        else:
            conn.execute('UPDATE users SET name = ? WHERE id = ?', 
                         (new_name, session['user_id']))

        conn.commit()
        conn.close()

        session['user_name'] = new_name
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('welcome'))

    conn.close()
    return render_template('edit_profile.html', user=user)

# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part in request.', 'danger')
#         return redirect(url_for('welcome'))
#     file = request.files['file']
#     if file.filename == '':
#         flash('No file selected.', 'danger')
#         return redirect(url_for('welcome'))
#     if file:
#         file_stream = file.read()
#         npimg = np.frombuffer(file_stream, np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#         if img is None:
#             flash('Invalid image file.', 'danger')
#             return redirect(url_for('welcome'))
#         annotated_img = predict_emotion_from_image(img)
#         _, buffer = cv2.imencode('.jpg', annotated_img)
#         io_buf = io.BytesIO(buffer)
#         return Response(io_buf.getvalue(), mimetype='image/jpeg')



# @app.route('/live_webcam', methods=['POST'])
# def live_webcam():
#     if 'frame' not in request.files:
#         return Response("No frame data", status=400)
#     frame_file = request.files['frame']
#     frame_stream = frame_file.read()
#     npimg = np.frombuffer(frame_stream, np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     if frame is None:
#         return Response("Invalid frame data", status=400)
#     annotated_frame = predict_emotion_from_image(frame)
#     _, buffer = cv2.imencode('.jpg', annotated_frame)
#     return Response(buffer.tobytes(), mimetype='image/jpeg')


# ----------------------------- second trial ------------------------------------------------


# @app.route('/live_webcam', methods=['POST'])
# def live_webcam():
#     if 'frame' not in request.files:
#         return Response("No frame data", status=400)
#     frame_file = request.files['frame']
#     frame_stream = frame_file.read()
#     npimg = np.frombuffer(frame_stream, np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     if frame is None:
#         return Response("Invalid frame data", status=400)
    
#     # Unpack the tuple returned by predict_emotion_from_image
#     annotated_frame, emotion_id = predict_emotion_from_image(frame)
    
#     # Now, annotated_frame is a valid image (numpy array)
#     _, buffer = cv2.imencode('.jpg', annotated_frame)
#     return Response(buffer.tobytes(), mimetype='image/jpeg')



# -----------------------------------------third trail ----------------------------------------------------------------





@app.route('/live_summary')
def live_summary():
    emotion_counts = session.get('emotion_counts')
    if not emotion_counts:
        return jsonify({"error": "No emotion data available."}), 400

    # Convert keys from string to int for processing
    emotion_counts_int = {int(k): v for k, v in emotion_counts.items()}
    
    # Import the recommendation function and emotion mapping from api.py
    from api import recommend_music, emotion_dict
    
    # Remap the counts to show emotion strings instead of numeric indices
    mapped_counts = {emotion_dict[k]: v for k, v in emotion_counts_int.items()}
    
    # Determine the emotion with the highest count
    highest_emotion_index = max(emotion_counts_int, key=emotion_counts_int.get)
    
    return jsonify({
        "emotion_counts": mapped_counts,
        "highest_emotion": emotion_dict[highest_emotion_index],
        "tracks": recommend_music(highest_emotion_index)
    })




# @app.route('/recommend_music')
# def recommend_music_route():
#     emotion_id = session.get('predicted_emotion')
#     if emotion_id is None:
#         return jsonify({"error": "No emotion detected"}), 400
#     tracks = recommend_music(emotion_id)
#     if not tracks:
#         return jsonify({"error": "No recommendations found"}), 500
#     return jsonify({"emotion": emotion_dict[emotion_id], "tracks": tracks})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part in request.', 'danger')
        return redirect(url_for('welcome'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('welcome'))
    if file:
        file_stream = file.read()
        npimg = np.frombuffer(file_stream, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            flash('Invalid image file.', 'danger')
            return redirect(url_for('welcome'))
        annotated_img, emotion_id = predict_emotion_from_image(img)
        # Store the predicted emotion in session for later use
        session['predicted_emotion'] = emotion_id
        _, buffer = cv2.imencode('.jpg', annotated_img)
        io_buf = io.BytesIO(buffer)
        return Response(io_buf.getvalue(), mimetype='image/jpeg')

@app.route('/live_webcam', methods=['POST'])
def live_webcam():
    if 'frame' not in request.files:
        return Response("No frame data", status=400)
    frame_file = request.files['frame']
    frame_stream = frame_file.read()
    npimg = np.frombuffer(frame_stream, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return Response("Invalid frame data", status=400)
    
    # Process the frame to get the annotated image and detected emotion id
    annotated_frame, emotion_id = predict_emotion_from_image(frame)
    
    # Initialize emotion_counts in session if not present
    if 'emotion_counts' not in session:
        # Assuming emotions are 0-6
        session['emotion_counts'] = {str(k): 0 for k in range(7)}
    
    # Update the count for the detected emotion (if valid)
    if emotion_id is not None:
        session['emotion_counts'][str(emotion_id)] += 1
        session.modified = True  # Ensure session changes are saved

    # (Optional) Print for debugging purposes
    print("Updated emotion_counts:", session.get('emotion_counts'))
    
    # Encode the annotated image to return it
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/recommend_music')
def recommend_music_route():
    emotion_id = session.get('predicted_emotion')
    if emotion_id is None:
        return jsonify({"error": "No emotion detected"}), 400
    from api import recommend_music, emotion_dict  # Ensure these are imported
    tracks = recommend_music(emotion_id)
    if not tracks:
        return jsonify({"error": "No recommendations found"}), 500
    return jsonify({"emotion": emotion_dict[emotion_id], "tracks": tracks})


if __name__ == '__main__':
    app.run(debug=True)