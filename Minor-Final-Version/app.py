import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from flask import Flask, render_template, Response
import sqlite3
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load the trained CNN model
model = load_model('models/model.keras')

# Initialize the attendance database
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()

init_db()

# Define the labels for students
student_labels = {
    0: "Vaishnavi",
    1: "Khushi",
    2: "Will Smith"
}

# Function to mark attendance in the database
def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check if the person already has a record for today
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND timestamp LIKE ?", (name, f"{current_date}%"))
    record = cursor.fetchone()
    
    # If no record exists, add a new entry
    if not record:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO attendance (name, status, timestamp) VALUES (?, ?, ?)", (name, 'Present', timestamp))
        conn.commit()
    
    conn.close()

# Capture video from the webcam
def gen_frames():
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (50, 50)).reshape(1, 50, 50, 1).astype('float32') / 255.0

            # Predict the face using the trained model
            model_out = model.predict(face)[0]
            label_index = np.argmax(model_out)
            label = student_labels.get(label_index)

            # If the label is recognized, mark attendance
            if label:
                mark_attendance(label)

                # Display the attendance on the frame
                cv2.putText(frame, f'{label} Present', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to view the attendance records
@app.route('/attendance')
def view_attendance():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, status, timestamp FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()

    return render_template('attendance.html', records=records)

# Home route to show the live video stream and attendance status
@app.route('/')
def index():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT name FROM attendance")
    records = cursor.fetchall()
    conn.close()

    attendance_record = [record[0] for record in records]

    return render_template('index.html', attendance_record=attendance_record)

if __name__ == "__main__":
    app.run(debug=True)
