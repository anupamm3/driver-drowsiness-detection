@workspace /new 

# Project Specification: Real-Time Driver Drowsiness Detection System

## 1. Role & Objective
Act as a Senior Computer Vision Architect. Your task is to implement a complete, deployment-ready Driver Drowsiness Detection System in Python. The system must use a "Hybrid Architecture" combining **MediaPipe Face Mesh** (for landmarks/geometry) and a **Custom CNN** (for eye state classification).

## 2. Tech Stack & Dependencies
* **Language:** Python 3.9+
* **Core Libraries:** `opencv-python`, `tensorflow`, `mediapipe`, `numpy`, `streamlit`, `pygame` (for audio alarm).
* **Environment:** Create a `requirements.txt` file immediately.

## 3. Directory Structure
Implement the following file structure exactly:
* `data/` (Empty folders for `raw`, `train/open`, `train/closed`)
* `src/`
    * `__init__.py`
    * `data_loader.py` (Script to parse MRL dataset)
    * `model_arch.py` (The CNN architecture definition)
    * `detector.py` (The main inference class connecting MediaPipe + CNN)
* `assets/` (Place a placeholder `alarm.wav` here)
* `models/` (Where the `.h5` file will be saved)
* `train_model.py` (Script to run the training loop)
* `app.py` (The Streamlit web application)
* `README.md` (Documentation)

## 4. Implementation Details (Step-by-Step)

### Phase A: Data Preparation (`src/data_loader.py`)
* **Task:** The user has already downloaded a pre-sorted version of the MRL dataset into `mrl_dataset_raw/`.
* **Source Structure:**
    * `mrl_dataset_raw/Open-Eyes/` (Contains ~40k open eye images)
    * `mrl_dataset_raw/Close-Eyes/` (Contains ~40k closed eye images)
* **Logic:** Write a function `organize_data()`.
    * **Action:** It must select a **random sample of 2,000 images** from the `Open-Eyes` folder and copy them to `data/train/open`.
    * **Action:** It must select a **random sample of 2,000 images** from the `Close-Eyes` folder and copy them to `data/train/closed`.
    * **Constraint:** Do not just take the first 2000; shuffle them first to ensure variety.
    * Print "Data setup complete" when finished.

### Phase B: The Model (`src/model_arch.py` & `train_model.py`)
* **Architecture:** Create a `build_model()` function returning a Sequential Keras model.
    * Input: `(64, 64, 1)` (Grayscale for speed).
    * Layers: 3x Conv2D (filters 32, 64, 128) + MaxPooling + Dropout(0.25) + Flatten + Dense(64) + Dense(2, softmax).
* **Training Script:**
    * Use `ImageDataGenerator` with validation split (0.2).
    * Compile with Adam optimizer.
    * Save model to `models/drowsiness_cnn.h5`.
    * Plot accuracy/loss graphs and save them as `assets/training_graph.png`.

### Phase C: The Inference Logic (`src/detector.py`)
* **Class:** `DrowsinessDetector`
* **Init:** Load the trained model and initialize MediaPipe Face Mesh (`refine_landmarks=True`).
* **Method `process_frame(frame)`:**
    1.  Convert frame to RGB.
    2.  Process with MediaPipe to get landmarks.
    3.  **Eye Logic (CNN):**
        * Extract coordinates for Left Eye and Right Eye.
        * Crop the eye regions, convert to Grayscale, resize to `(64, 64)`.
        * Run model prediction.
        * If `prob(Closed) > 0.9` for BOTH eyes, increment `CLOSED_FRAMES` counter.
    4.  **Mouth Logic (Geometry):**
        * Extract Upper/Lower lip landmarks.
        * Calculate `MAR` (Mouth Aspect Ratio).
        * If `MAR > 0.5`, increment `YAWN_FRAMES` counter.
    5.  **Return:** A dictionary with `{'left_eye': 'Open/Closed', 'score': int, 'yawn': bool, 'annotated_frame': frame}`.

### Phase D: The UI (`app.py`)
* **Framework:** Streamlit.
* **Features:**
    * Sidebar: Settings (Sensitivity Thresholds).
    * Main: Real-time video feed using `cv2.VideoCapture` loops (use `st.image` as a placeholder for the video frame).
    * **Logic:**
        * If `detector.score > 15`: Show **"DROWSY ALERT!"** in Red text and play alarm.
        * If `detector.yawn`: Show **"Yawning..."** warning.
    * **Stats:** Use `st.line_chart` to plot the "Drowsiness Score" history in real-time.

## 5. Execution Instructions
* Generate the code for all files listed above.
* Ensure the code handles "No Face Detected" errors gracefully (don't crash).
* Add extensive comments explaining the code for a student interview context.