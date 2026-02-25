# 🏍️ Helmet Violation Detection System

AI-powered system for detecting helmet violations and reading license plates in real-time.

## Features
- 🎥 Live webcam detection
- 📤 Image upload detection
- 🔢 License plate OCR
- 💾 Auto-save violations to Excel
- 📊 Violation history tracking

## Technologies
- YOLOv8 for object detection
- PaddleOCR for license plate reading
- Streamlit for web interface

## Live Demo
[View App](your-app-url-here)

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Models
Place your trained models in the `models/` directory:
- `best_helmet_model.pt` - Helmet detection
- `best.pt` - License plate detection