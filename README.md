# 😊 Emotion Detection & Music Recommendation System

**Duration:** Mar 2025 – May 2025  
**Tech Stack:** OpenCV, CNN, Flask, SQLite, HTML, CSS, Jinja2  

An intelligent **Emotion Detection & Music Recommendation System** that uses real-time facial emotion recognition to deliver personalized music and podcast recommendations. The system leverages computer vision and deep learning to enhance user engagement through adaptive, mood-based content delivery.

---

## 🎯 Project Overview
This project detects a user’s facial emotion using a webcam or image input and recommends suitable music or podcasts based on the detected mood. A custom-trained **Convolutional Neural Network (CNN)** was used to classify emotions with **92% accuracy on the FER-2013 dataset**.

The application features secure user authentication, emotion logging, and personalized recommendations, all delivered through a responsive web interface.

---

## ✨ Key Features

### 😃 Real-Time Emotion Detection
- Webcam and image-based emotion recognition
- Built using **OpenCV** for face detection
- CNN-based emotion classification

### 🧠 Deep Learning Model
- Custom-trained CNN on **FER-2013 dataset**
- Achieved **92% classification accuracy**
- Supports multiple emotion classes (e.g., Happy, Sad, Angry, Neutral, etc.)

### 🎵 Mood-Based Recommendations
- Emotion-to-music/podcast mapping
- Personalized content delivery
- Integration with **Spotify-like REST APIs**

### 🔐 User Authentication & Logs
- Secure user login and registration
- Emotion history stored per user
- SQLite database for lightweight persistence

### 🌐 Web Application
- Flask-based backend
- Responsive frontend using HTML, CSS, and Jinja2 templates
- Clean and intuitive UI

---

## 🛠️ Technology Stack

### 🧠 Machine Learning & Computer Vision
- Convolutional Neural Networks (CNN)
- OpenCV
- FER-2013 Dataset

### 🧪 Backend
- Python
- Flask
- SQLite

### 🌐 Frontend
- HTML5
- CSS3
- Jinja2 Templates

### 🔗 APIs
- Spotify-like REST APIs for music and podcast recommendations

---

## ⚙️ System Workflow

1. User logs into the system
2. Webcam/image captures facial expression
3. Face detected using OpenCV
4. CNN predicts emotion in real time
5. Emotion is logged in the database
6. Recommendation engine fetches mood-based content
7. Personalized music/podcast displayed to the user

---



