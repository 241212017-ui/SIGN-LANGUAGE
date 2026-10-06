# Real-Time Sign Language Recognition

A real-time sign language recognition application that detects hand gestures from a live webcam feed and identifies the corresponding English alphabet letters from A to Z. Built with Python, OpenCV, PyTorch, and a pretrained SigLIP vision model, this project demonstrates how artificial intelligence can support accessibility and inclusive communication.

## Overview

This project is designed to interpret sign language gestures in real time using computer vision and deep learning. The application captures video from the camera, processes each frame, and predicts the most likely letter based on the user’s hand position and shape. The prediction is shown directly in the interface along with a confidence score, making the system both practical and easy to understand.

The solution is implemented as a lightweight Streamlit web app, allowing users to run it locally with minimal setup. It is well-suited for educational demos, accessibility research, and future expansion into full word and sentence recognition.

## Key Features

- Live webcam-based sign detection
- Real-time recognition for alphabet gestures from A to Z
- Confidence score for each prediction
- User-friendly interface built with Streamlit
- Automatic device handling for CPU or GPU
- Easily extensible for advanced sign language recognition workflows

## Technology Stack

- Python
- Streamlit
- OpenCV
- PyTorch
- Hugging Face Transformers
- Pillow
- NumPy
- streamlit-webrtc

## How It Works

1. The webcam captures a live video frame.
2. The frame is processed and converted into a format suitable for the model.
3. A pretrained SigLIP image classifier predicts the most likely hand-sign class.
4. The application displays the recognized letter and confidence percentage on the video feed.

## Project Goals

This project aims to promote accessibility by making communication easier for deaf and hard-of-hearing individuals. By combining live computer vision with AI-powered recognition, it demonstrates the potential of technology to reduce communication barriers and support inclusive experiences in everyday life.

## Installation

### Prerequisites

- Python 3.10+
- A working webcam
- pip or conda environment

### Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

Run the application and allow access to the webcam. The app will begin recognizing sign language gestures live and display the predicted letter in the interface.

## Future Enhancements

- Expand from single-letter detection to full-word recognition
- Add phrase and sentence generation
- Improve accuracy with larger training datasets
- Support multiple sign language styles and regional variations
- Add model tuning for better real-world performance

## License

This project is intended for educational and research purposes.
