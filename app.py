import streamlit as st
import cv2
import numpy as np
from src.detector import DrowsinessDetector

# Initialize the drowsiness detector
detector = DrowsinessDetector(model_path='models/drowsiness_cnn.h5')

# Sidebar for settings
st.sidebar.title("Settings")
sensitivity_threshold = st.sidebar.slider("Sensitivity Threshold", 1, 50, 15)

# Placeholder for video feed
st.title("Real-Time Driver Drowsiness Detection")
frame_placeholder = st.empty()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Store drowsiness score history
drowsiness_scores = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with the detector
    result = detector.process_frame(frame)

    # Update drowsiness score history
    drowsiness_scores.append(result['score'])

    # Display the annotated frame
    frame_placeholder.image(cv2.cvtColor(result['annotated_frame'], cv2.COLOR_BGR2RGB), channels="RGB")

    # Check for drowsiness alert
    if result['score'] > sensitivity_threshold:
        st.warning("DROWSY ALERT!", icon="⚠️")
        # Play alarm (placeholder logic)
        # pygame.mixer.init()
        # pygame.mixer.music.load('assets/alarm.wav')
        # pygame.mixer.music.play()

    # Check for yawning alert
    if result['yawn']:
        st.warning("Yawning...", icon="😴")

    # Display drowsiness score history
    st.line_chart(drowsiness_scores)

# Release the video capture
cap.release()