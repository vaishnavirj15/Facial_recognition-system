# Facial Recognition System

This project is a Flask-based web application that automates attendance management using facial recognition technology. It captures and recognizes faces in real-time to mark attendance, improving efficiency, accuracy, and eliminating manual errors.

---

## Features
- **Real-time Face Detection**: Detects and identifies faces using a webcam or uploaded image input.
- **Automated Attendance Marking**: Records attendance automatically upon successful face recognition.
- **Database Integration**: Stores attendance logs in a secure database (SQLite or MySQL).
- **User-Friendly Interface**: Simple and intuitive web interface built with Flask.
- **Scalability**: Supports multiple users and can be extended with features like reports and analytics.

---

## Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Face Recognition Library**: OpenCV, tensorflow,keras,scikit-learn(Python libraries)
- **Database**: SQLite or MySQL
- **Deployment**: Local Server

---

## Setup Instructions

### **Pre-requisites**
- Python 3.7 or higher installed on your system.
- Install required libraries and tools (details below).

---

### **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/facial-recognition-system.git
   cd facial-recognition-system
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Add Facial Data**:
   - **Add your own facial data into the data folder**.
   - **Edit your own label in the code and data**.
---

### **Running the Application**

1. **Start the Flask Server**:
   ```bash
   python app.py
2. **Access the Web Interface**:
   - **Open your browser and go to: http://127.0.0.1:5000/**
---


### **Notes**
- **Ensure the data folder contains labeled facial data for recognition**.
- **The system currently supports 3 labels but can be expanded as needed**.
- **The system uses haar-cascade-classifier for initial face detection**.
---

### Key Highlights:
1. **Professional Layout**: Includes features, tech stack, and clear setup instructions.
2. **Customizable Options**: Describes how to expand the system for additional labels or features.
3. **User-Friendly**: Simplified instructions for both technical and non-technical users.

Replace `https://github.com/your-username/facial-recognition-system.git` with your actual GitHub repository URL. Let me know if you'd like further enhancements!

