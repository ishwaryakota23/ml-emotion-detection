# 😄 Emotion Detection Web App

A user-friendly Emotion Detection web application that uses deep learning (CNN with Keras) to identify human emotions from facial images. It offers a sleek UI with both image upload and webcam support, emoji-based emotion display, and dark mode toggle.

---

## 🔍 Project Overview

This system detects basic human emotions from facial images using a Convolutional Neural Network (CNN) model built with Keras. It features:
- Image upload (drag-and-drop or file picker)
- Real-time webcam capture
- Predicted emotion display with emoji
- Dark mode toggle for better UX
- Reset functionality

---

## 🛠️ Features

- 🎯 CNN-based Emotion Detection (`final_emotion_model.h5`)
- 📷 Upload image or use webcam
- 😀 Emoji-based emotion output
- 🌗 Light/Dark mode toggle
- 🔄 Reset to clear previous results
- 💻 Fully responsive HTML/CSS/JS frontend
- 🧠 Flask backend for prediction

---

## 🧱 Technologies Used

### Frontend:
- HTML5
- CSS3 (Responsive Design)
- JavaScript (FormData, Fetch API)

### Backend:
- Python (Flask)
- TensorFlow / Keras (CNN model)
- OpenCV (for webcam capture if integrated)
- NumPy, Pillow

---


## 🧠 Model Overview

- Model Type: Convolutional Neural Network (CNN)
- Framework: Keras with TensorFlow backend
- Input: Facial image
- Output: One of the following emotions:
  - Happy
  - Sad
  - Angry
  - Surprised
  - Neutral
  - (Extendable with more classes)

---


