# Dashcam Collision Prediction

A deep learning project for predicting **vehicle collisions from dashcam video footage** using **Computer Vision, CNNs, and LSTM-based sequence modeling**.

This project was developed using the **Nexar Dashcam Collision Prediction Challenge** dataset on Kaggle, where the objective is to identify dangerous driving situations and predict collisions as early as possible.

---

## Problem Statement

Road accidents can often be anticipated shortly before they occur. The goal of this project is to use dashcam video footage to learn visual and temporal patterns that indicate an upcoming collision.

The Nexar dataset contains three types of driving scenarios:

- **Collision** – an accident occurs
- **Near-Miss** – a dangerous situation occurs but an accident is avoided
- **Non-Collision** – normal driving

The dataset also provides temporal annotations such as **event time** and **alert time**, representing when the event occurs and when it becomes predictable.

---

## Project Objective

The project aims to:

1. Process dashcam videos into usable image sequences.
2. Extract and preprocess video frames.
3. Extract spatial features using a Convolutional Neural Network.
4. Capture temporal relationships between consecutive frames using an LSTM.
5. Predict:
   - **Event Time** – when the collision occurs.
   - **Alert Time** – when the collision could first be predicted.

---

## Model Architecture

The project combines **CNN-based spatial feature extraction** with **LSTM-based temporal modeling**.

```text
Dashcam Video
      │
      ▼
Frame Extraction
      │
      ▼
Frame Resizing & Normalization
      │
      ▼
TimeDistributed CNN
      │
      ├── Conv2D
      ├── MaxPooling
      ├── Conv2D
      ├── MaxPooling
      └── Flatten
      │
      ▼
     LSTM
   (128 units)
      │
      ▼
 Dense Layer
    (64)
      │
      ▼
   Dropout
      │
      ▼
 Output Layer
 ┌───────────────┐
 │  Event Time   │
 │  Alert Time   │
 └───────────────┘
```

## Project Pipeline

### 1. Dataset Organization

The dataset is organized into separate directories for crash and non-crash videos.
Dataset metadata is processed using Pandas, and videos are categorized according to their labels.

### 2. Video & Frame Processing

Videos are processed using OpenCV.

For each video:

- Frames are extracted from the video.
- Frames are resized to 112 x 112.
- Pixel values are preprocessed/normalized.
- A fixed-length sequence of frames is created.
- Missing frames are padded when required.

For collision videos, the extraction process focuses on the period leading up to the recorded event.

### 3. Data Preparation

The extracted video frame sequences are converted into NumPy arrays.

The dataset is divided into:

- 80% Training
- 20% Validation

using `train_test_split` with a fixed random state for reproducibility.

---

## CNN-LSTM Model

The model uses a combination of convolutional and recurrent layers.

### CNN Component

The CNN extracts spatial features from individual video frames using:

- Conv2D
- MaxPooling2D
- Flatten

The CNN is wrapped using `TimeDistributed` so that the same feature extractor is applied to each frame in the sequence.

### LSTM Component

The extracted frame-level features are passed to an:

- `LSTM(128)`

The LSTM learns temporal relationships between consecutive frames and helps identify patterns that may indicate an upcoming collision.

### Output

The model predicts two continuous values:

- Event Time
- Alert Time

---

## Training Configuration

| Parameter | Value |
|---|---|
| Frame Size | 112 x 112 |
| Channels | 3 |
| Sequence Length | 6 |
| Batch Size | 8 |
| Epochs | 10 |
| Learning Rate | 1e-5 |
| Optimizer | Adam |
| Loss Function | Mean Absolute Error |
| Validation Split | 20% |
| Temporal Model | LSTM |
| Feature Extractor | CNN |

Training also uses techniques such as:

- Early Stopping
- Learning Rate Reduction
- Model Checkpointing

---

## Prediction & Inference

The trained model can be used to perform predictions on individual dashcam videos.

The inference pipeline:

1. Loads the video.
2. Extracts the required frames.
3. Resizes and preprocesses the frames.
4. Creates the required frame sequence.
5. Loads the trained CNN-LSTM model.
6. Generates predictions.
7. Converts the predicted values back into time-based values.

The predicted event time and alert time can then be compared with the corresponding ground-truth values.

---

### Script Description

**1_Dataset_Structure.py**
Organizes the downloaded videos into appropriate directories based on their labels.

**2_Frame_Extraction.py**
Uses OpenCV to extract and preprocess frames from the dashcam videos and prepares them as NumPy sequences.

**3_Model_Training.py**
Builds and trains the CNN-LSTM deep learning model using the extracted frame sequences.

**4_Prediction_and_Inference.py**
Loads the trained model and performs inference on selected videos to predict event and alert times.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/AbhinavPamadi/Collison_Prediction_Dashcam.git
cd Collison_Prediction_Dashcam
```

### 2. Install Dependencies

```bash
pip install numpy pandas opencv-python tensorflow scikit-learn tqdm
```

### 3. Download the Dataset

Download the Nexar Collision Prediction dataset from Kaggle:
https://www.kaggle.com/competitions/nexar-collision-prediction/data
Update the dataset paths in the Python scripts according to your local environment.

### 4. Organize the Dataset

```bash
python 1_Dataset_Structure.py
```

### 5. Extract Video Frames

```bash
python 2_Frame_Extraction.py
```

### 6. Train the Model

```bash
python 3_Model_Training.py
```

### 7. Run Inference

```bash
python 4_Prediction_and_Inference.py
```

---
