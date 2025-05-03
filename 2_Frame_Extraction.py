import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tqdm import tqdm

# CONFIGURATION
DATASET_DIR      = r"C:\Users\..\Documents\collision_prediction_dataset\dataset"
CSV_PATH         = r"C:\Users\..\Documents\collision_prediction_dataset\train.csv"
OUTPUT_FRAME_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\frames"
SEQUENCE_LENGTH  = 10
FRAME_HEIGHT     = 112
FRAME_WIDTH      = 112

# Load CSV labels
df = pd.read_csv(CSV_PATH)
os.makedirs(OUTPUT_FRAME_DIR, exist_ok=True)

subfolders = ['Crash', 'NonCrash']
def find_video_path(vid):
    for sub in subfolders:
        path = os.path.join(DATASET_DIR, sub, f"{vid}.mp4")
        if os.path.exists(path):
            return path, sub
    return None, None

# Extract frames function
def extract_frames(video_path, start_time_sec=0.0, max_frames=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)
    cap.release()
    while len(frames) < max_frames:
        frames.append(frames[-1] if frames else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32))
    return np.array(frames)

processed = 0
for _, row in tqdm(df.iterrows(), total=len(df)):
    vid = str(int(row['id'])).zfill(5)
    video_path, label = find_video_path(vid)
    if not video_path:
        # Video not found in any subfolder
        continue
    # Determine start time
    if row['target'] == 1 and not pd.isna(row['time_of_event']):
        start_sec = max(0, row['time_of_event'] - 5)
    else:
        start_sec = 0.0
    # Extract
    frames = extract_frames(video_path, start_sec)
    
    out_dir = os.path.join(OUTPUT_FRAME_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{vid}.npy"), frames)
    processed += 1

print(f" Frame extraction complete. {processed} videos processed.")
