# --- PART 4: SINGLE VIDEO INFERENCE ---
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


VIDEO_ID = 1924
MODEL_PATH = r"C:\Users\..\Documents\collision_prediction_dataset\collision_time_predictor_norm.keras"
FRAME_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\frames"
CSV_PATH = r"C:\Users\..\Documents\collision_prediction_dataset\train.csv"
SEQUENCE_LENGTH = 20
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3
MAX_DURATION = 40.0


model = load_model(MODEL_PATH)


vid = str(int(VIDEO_ID)).zfill(5)
df = pd.read_csv(CSV_PATH)
row = df[df['id'] == VIDEO_ID].iloc[0]

# Check path
path = os.path.join(FRAME_DIR, "Crash", f"{vid}.npy")
if not os.path.exists(path):
    print(f" No crash in {vid} video")
    exit()

# Load and preprocess
seq = np.load(path)
seq = seq[:SEQUENCE_LENGTH]
frames = [cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT)) for f in seq]
if len(frames) < SEQUENCE_LENGTH:
    frames += [frames[-1]] * (SEQUENCE_LENGTH - len(frames))
arr = np.array(frames, dtype=np.float32) / 255.0
arr = np.expand_dims(arr, axis=0)

# Predict
pred = model.predict(arr)[0]
pred_event = round(pred[0] * MAX_DURATION, 2)
pred_alert = round(pred[1] * MAX_DURATION, 2)

print(f"  Prediction for Video ID {VIDEO_ID}")
print(f"  True Event Time: {row['time_of_event']}")
print(f"  True Alert Time: {row['time_of_alert']}")
print(f"  Predicted Event Time: {pred_event}")
print(f"  Predicted Alert Time: {pred_alert}")
