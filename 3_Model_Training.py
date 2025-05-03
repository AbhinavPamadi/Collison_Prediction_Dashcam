import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from glob import glob
import random

# CONFIGURATION 
FRAME_DIR = r"C:\Users\..\Documents\collision_prediction_dataset\frames"
SEQUENCE_LENGTH = 6      
FRAME_HEIGHT = 112       
FRAME_WIDTH = 112        
CHANNELS = 3             
EPOCHS = 10              
BATCH_SIZE = 8           
MODEL_SAVE_PATH = r"C:\Users\..\Documents\collision_prediction_dataset\car_crash_detector_model.keras"

# DATA GENERATOR 
class FrameSequenceGenerator(Sequence):
    def __init__(self, filepaths, labels, batch_size):
        self.filepaths = filepaths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.filepaths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = []
        for path in batch_paths:
            frames = np.load(path)[:SEQUENCE_LENGTH]
            frames = np.array([cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT)) for f in frames])
            if len(frames) < SEQUENCE_LENGTH:
                pad = np.tile(frames[-1:], (SEQUENCE_LENGTH - len(frames), 1, 1, 1))
                frames = np.concatenate([frames, pad], axis=0)
            batch_data.append(frames)
        return np.array(batch_data), np.array(batch_labels)

# GATHER FILEPATHS AND LABELS 
crash_paths = glob(os.path.join(FRAME_DIR, "Crash", "*.npy"))
noncrash_paths = glob(os.path.join(FRAME_DIR, "NonCrash", "*.npy"))
filepaths = crash_paths + noncrash_paths
labels = [1] * len(crash_paths) + [0] * len(noncrash_paths)

# SHUFFLE AND SPLIT
combined = list(zip(filepaths, labels))
random.shuffle(combined)
filepaths, labels = zip(*combined)
split_idx = int(0.8 * len(filepaths))
train_paths, val_paths = filepaths[:split_idx], filepaths[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]

# GENERATORS 
train_gen = FrameSequenceGenerator(train_paths, train_labels, BATCH_SIZE)
val_gen = FrameSequenceGenerator(val_paths, val_labels, BATCH_SIZE)

# MODEL ARCHITECTURE
input_layer = Input(shape=(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_layer)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
x = TimeDistributed(GlobalAveragePooling2D())(x)
x = LSTM(64)(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# CALLBACKS
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# TRAIN 
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)


model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")


loss, acc = model.evaluate(val_gen)
print(f"✅ Test Accuracy: {acc:.4f}")

# PLOT LOSS & ACCURACY 
plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# CONFUSION MATRIX 
y_true, y_pred = [], []
for Xb, yb in val_gen:
    preds = (model.predict(Xb) >= 0.5).astype(int).flatten()
    y_true.extend(yb)
    y_pred.extend(preds)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NonCrash','Crash'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()
