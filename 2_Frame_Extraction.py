import os
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tqdm import tqdm

# CONFIGURATION 
VIDEO_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\dataset"
OUTPUT_FRAME_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\frames"
FRAME_HEIGHT = 299
FRAME_WIDTH = 299
SEQUENCE_LENGTH = 10  # Number of frames per video

# CREATE FRAME OUTPUT DIRECTORY
os.makedirs(OUTPUT_FRAME_DIR, exist_ok=True)

# FUNCTION TO EXTRACT FRAMES 
def extract_frames(video_path, max_frames=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  
        frame = preprocess_input(frame.astype(np.float32))  
        frames.append(frame)
    cap.release()

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])  
    return np.array(frames[:SEQUENCE_LENGTH])

# EXTRACT FRAMES FROM VIDEOS IN BOTH CRASH & NONCRASH FOLDERS
for label in ["Crash", "NonCrash"]:
    label_dir = os.path.join(VIDEO_DIR, label)
    target_frame_dir = os.path.join(OUTPUT_FRAME_DIR, label)
    os.makedirs(target_frame_dir, exist_ok=True)

    for video_file in tqdm(os.listdir(label_dir)):
        video_path = os.path.join(label_dir, video_file)
        
        if video_file.endswith(".mp4"):
            frames = extract_frames(video_path)
           
            np.save(os.path.join(target_frame_dir, f"{os.path.splitext(video_file)[0]}.npy"), frames)

print("Frames extracted and saved in respective folders.")
