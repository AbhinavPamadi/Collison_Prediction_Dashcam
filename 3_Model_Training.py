import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# CONFIGURATION
FRAME_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\frames"
CSV_PATH = r"C:\Users\..\Documents\collision_prediction_dataset\train.csv"
SEQUENCE_LENGTH = 6  
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-5

# LOAD CSV
df = pd.read_csv(CSV_PATH)


filepaths, targets = [], []
for _, row in df.iterrows():
    vid = str(int(row['id'])).zfill(5)
    for label in ['Crash', 'NonCrash']:
        path = os.path.join(FRAME_DIR, label, f"{vid}.npy")
        if os.path.exists(path):
            filepaths.append(path)
            if label == 'Crash' and not pd.isna(row['time_of_event']):
                evt = float(row['time_of_event']) 
                alt = float(row['time_of_alert']) 
                targets.append([evt, alt])
            else:
                targets.append([0.0, 0.0])  
            break

print(f"Total samples: {len(filepaths)} (Crash+NonCrash)")
if len(filepaths) == 0:
    raise RuntimeError("No .npy frames found. Run Part 2 extraction first.")

# SPLIT
train_paths, val_paths, train_targets, val_targets = train_test_split(
    filepaths, targets, test_size=0.2, random_state=42
)
print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

# DATA GENERATOR
def data_generator(paths, targets):
    while True:
        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i:i+BATCH_SIZE]
            batch_y = targets[i:i+BATCH_SIZE]
            batch_X = []
            for p in batch_paths:
                seq = np.load(p)
                seq = seq[:SEQUENCE_LENGTH]
                # resize and normalize
                frames = [cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT)) for f in seq]
                if len(frames) < SEQUENCE_LENGTH:
                    pad = [frames[-1]]*(SEQUENCE_LENGTH - len(frames))
                    frames.extend(pad)
                arr = np.array(frames, dtype=np.float32)/255.0
                batch_X.append(arr)
            yield np.array(batch_X), np.array(batch_y, dtype=np.float32)

# MODEL DEFINITION
model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)),
    TimeDistributed(MaxPooling2D(2,2)),
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D(2,2)),
    TimeDistributed(Flatten()),
    LSTM(128),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='linear')  
])
model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss='mae',
    metrics=['mae']
)
model.summary()

# CALLBACKS
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint("collision_time_predictor_norm.keras", save_best_only=True)
]

# TRAIN
model.fit(
    data_generator(train_paths, train_targets),
    validation_data=data_generator(val_paths, val_targets),
    steps_per_epoch=len(train_paths)//BATCH_SIZE,
    validation_steps=len(val_paths)//BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# SAVE FINAL
model.save(r"C:\Users\..\Documents\collision_prediction_dataset\collision_time_predictor_norm.keras")
print("Training complete and model saved as collision_time_predictor_norm.keras")
