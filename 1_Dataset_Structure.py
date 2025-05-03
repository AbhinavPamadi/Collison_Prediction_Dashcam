import os
import shutil
import pandas as pd

# CONFIGURATION 
CSV_PATH = r"C:\Users\..\Documents\collision_prediction_dataset\train.csv"
VIDEO_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\train"
OUTPUT_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\dataset"

df = pd.read_csv(CSV_PATH)

# CREATE FOLDER STRUCTURE
for _, row in df.iterrows():
    video = str(int(row['id'])).zfill(5) + ".mp4"

    label = row['target']

    target_dir = os.path.join(OUTPUT_DIR, "Crash" if label == 1 else "NonCrash")
    os.makedirs(target_dir, exist_ok=True)

    src = os.path.join(VIDEO_DIR, video)
    dst = os.path.join(target_dir, video)

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f" Missing: {src}") 

print(" Dataset organized into Crash/NonCrash folders.")
