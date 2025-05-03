import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# CONFIGURATION 
MODEL_PATH = r"C:\Users\..\Documents\collision_prediction_dataset\car_crash_detector_model.keras"
SEQUENCE_LENGTH = 6
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3

model = load_model(MODEL_PATH)

# FUNCTION TO EXTRACT FRAMES FROM VIDEO
def extract_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])

    return np.array(frames[:SEQUENCE_LENGTH])

# PREDICTION FUNCTION
def predict_collision(video_path):
    frames = extract_video_frames(video_path)
    input_data = np.expand_dims(frames, axis=0)  # shape: (1, 6, 112, 112, 3)
    prediction = model.predict(input_data)[0][0]
    label = "Crash" if prediction >= 0.5 else "NonCrash"
    print(f"Prediction: {label} (Confidence: {prediction:.4f})")
    return label, prediction


test_video_path = r"C:\Users\..\Documents\collision_prediction_dataset\test\00012.mp4"
predict_collision(test_video_path)
