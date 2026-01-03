import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

class DrowsinessDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" Model file not found: {model_path}\n")
        # Load CNN model
        try:
            self.model = load_model(model_path)
            print(f" CNN Model loaded successfully from: {model_path}")
        except Exception as e:
            raise Exception(f" Error loading CNN model: {e}")
        
        # Initialize MediaPipe Face Mesh
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(" MediaPipe Face Mesh initialized successfully")
        except Exception as e:
            raise Exception(f" Error initializing MediaPipe: {e}")
        
        # Eye landmark indices (MediaPipe uses 468 points)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Drowsiness tracking parameters
        self.CLOSED_FRAMES = 0
        self.DROWSINESS_THRESHOLD = 15
        self.EYE_CLOSED_THRESHOLD = 0.30
        
        self.EAR_THRESHOLD = 0.21

        # Temporal smoothing buffers (reduces flickering)
        self.eye_state_history = {
            'left': [],
            'right': []
        }
        self.HISTORY_SIZE = 5  # Look at last 5 frames for majority vote
        
        print(f" Settings: Closed Threshold={self.EYE_CLOSED_THRESHOLD}, History Size={self.HISTORY_SIZE}")
    
    def process_frame(self, frame):
        # Convert to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        # No face detected
        if not results.multi_face_landmarks:
            self.CLOSED_FRAMES = max(0, self.CLOSED_FRAMES - 1)
            return {
                'drowsy': False,
                'score': self.CLOSED_FRAMES,
                'left_eye_state': 'No Face',
                'right_eye_state': 'No Face',
                'left_confidence': 0.0,
                'right_confidence': 0.0,
                'face_detected': False,
                'annotated_frame': self._draw_no_face_warning(frame)
            }
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract eye landmarks
        left_eye = self._get_landmarks(face_landmarks, self.LEFT_EYE, w, h)
        right_eye = self._get_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
        
        # Predict eye states WITH temporal smoothing
        left_eye_state, left_confidence = self._predict_eye_state_smoothed(frame, left_eye, 'left')
        right_eye_state, right_confidence = self._predict_eye_state_smoothed(frame, right_eye, 'right')
        
        # Update closed frames counter with better logic
        if left_eye_state == 'Closed' and right_eye_state == 'Closed':
            self.CLOSED_FRAMES += 1
        elif left_eye_state == 'Open' and right_eye_state == 'Open':
            # Only decrement if BOTH eyes are clearly open
            self.CLOSED_FRAMES = max(0, self.CLOSED_FRAMES - 1)
        # If one closed, one open - keep counter same (ambiguous frame)
        
        is_drowsy = self.CLOSED_FRAMES >= self.DROWSINESS_THRESHOLD
        
        # Annotate frame
        annotated_frame = self._annotate_frame(
            frame, left_eye, right_eye,
            left_eye_state, right_eye_state,
            left_confidence, right_confidence,
            is_drowsy
        )
        
        return {
            'drowsy': is_drowsy,
            'score': self.CLOSED_FRAMES,
            'left_eye_state': left_eye_state,
            'right_eye_state': right_eye_state,
            'left_confidence': left_confidence,
            'right_confidence': right_confidence,
            'face_detected': True,
            'annotated_frame': annotated_frame
        }
    
    #Convert MediaPipe landmarks to pixel coordinates.
    def _get_landmarks(self, face_landmarks, indices, width, height):
        landmarks = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y])
        return np.array(landmarks, dtype=np.int32)
    
    # Calculate Eye Aspect Ratio (EAR) for geometric eye closure detection.
    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        # Horizontal distance (eye width)
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Vertical distances (eye height at two positions)
        vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Calculate EAR
        if horizontal == 0:
            return 0.3  # Default value to avoid division by zero
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        
        return ear
    
    # Predict eye state with temporal smoothing to reduce flickering.
    def _predict_eye_state_smoothed(self, frame, eye_landmarks, eye_side):
        # STEP 1: Calculate EAR (geometric method)
        ear = self._calculate_eye_aspect_ratio(eye_landmarks)
        
        # STEP 2: Get CNN prediction
        raw_state, raw_confidence = self._predict_eye_state(frame, eye_landmarks)
        
        # STEP 3: COMBINED DECISION (EAR + CNN)
        # If EAR indicates closed, override CNN if needed
        if ear < self.EAR_THRESHOLD:
            # Geometric analysis says CLOSED - more reliable for squinting!
            raw_state = 'Closed'
            raw_confidence = max(raw_confidence, 0.75)  # Boost confidence
        
        # Add to history buffer
        self.eye_state_history[eye_side].append(raw_state)
        if len(self.eye_state_history[eye_side]) > self.HISTORY_SIZE:
            self.eye_state_history[eye_side].pop(0)
        
        # If buffer is not full yet, return raw prediction
        if len(self.eye_state_history[eye_side]) < 3:
            return raw_state, raw_confidence
        
        # MAJORITY VOTE: Count 'Closed' votes in history
        closed_votes = self.eye_state_history[eye_side].count('Closed')
        total_votes = len(self.eye_state_history[eye_side])
        
        # If majority says closed, return closed
        if closed_votes > total_votes / 2:
            return 'Closed', raw_confidence
        else:
            return 'Open', raw_confidence
    
    # Predict eye state using CNN
    def _predict_eye_state(self, frame, eye_landmarks):
        eye_image = self._extract_eye_region(frame, eye_landmarks)
        
        if eye_image is None:
            return 'Unknown', 0.0
        
        # Get prediction from model
        prediction = self.model.predict(eye_image, verbose=0)
        
        # {'closed': 0, 'open': 1}
        closed_prob = prediction[0][0]
        open_prob = prediction[0][1]
        
        # IMPROVED DECISION LOGIC with confidence difference
        prob_diff = closed_prob - open_prob
        
        # Use adaptive threshold based on confidence difference
        if closed_prob > 0.25:
            # Definitely closed
            return 'Closed', closed_prob
        elif open_prob > 0.70:
            # Definitely open (higher threshold for open to reduce false negatives)
            return 'Open', open_prob
        elif prob_diff > 0.05:
            # Closed probability slightly higher
            return 'Closed', closed_prob
        else:
            # Default to open when ambiguous
            return 'Open', open_prob
    
    # Extract and preprocess eye region for CNN input with improvements.
    def _extract_eye_region(self, frame, eye_landmarks):
        x_coords = eye_landmarks[:, 0]
        y_coords = eye_landmarks[:, 1]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Calculate eye width and height
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        
        # INCREASED padding for squinting detection
        padding_x = int(eye_width * 0.6)   # ← INCREASED from 0.5 to 0.6
        padding_y = int(eye_height * 1.0)  # ← INCREASED from 0.8 to 1.0
        
        h, w = frame.shape[:2]
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        # Crop eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        if eye_region.shape[0] < 5 or eye_region.shape[1] < 5:
            return None
        
        # Convert to grayscale
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # STRONGER contrast enhancement
        eye_gray = cv2.equalizeHist(eye_gray)
        
        # ADDED: Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eye_gray = clahe.apply(eye_gray)
        
        # Slight blur
        eye_gray = cv2.GaussianBlur(eye_gray, (3, 3), 0)
        
        # Resize to 64x64
        eye_resized = cv2.resize(eye_gray, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        eye_normalized = eye_resized.astype('float32') / 255.0
        
        # Reshape for CNN
        eye_final = eye_normalized.reshape(1, 64, 64, 1)
        
        return eye_final
    
    # Draw annotations on the frame with improved visualization.
    def _annotate_frame(self, frame, left_eye, right_eye, left_state, right_state, left_conf, right_conf, is_drowsy):
        annotated = frame.copy()
        
        # Draw eye contours with color coding
        left_color = (0, 0, 255) if left_state == 'Closed' else (0, 255, 0)
        right_color = (0, 0, 255) if right_state == 'Closed' else (0, 255, 0)
        
        # Thicker lines for better visibility
        cv2.polylines(annotated, [left_eye], True, left_color, 3)
        cv2.polylines(annotated, [right_eye], True, right_color, 3)
        
        # Draw labels with confidence scores
        left_center = left_eye.mean(axis=0).astype(int)
        right_center = right_eye.mean(axis=0).astype(int)
        
        # Add background rectangles for better text visibility
        left_text = f"{left_state} ({left_conf:.2f})"
        right_text = f"{right_state} ({right_conf:.2f})"
        
        # Left eye label
        left_label_pos = tuple(left_center - [0, 25])
        cv2.rectangle(annotated, 
                    (left_label_pos[0] - 5, left_label_pos[1] - 20),
                    (left_label_pos[0] + 100, left_label_pos[1] + 5),
                    (0, 0, 0), -1)
        cv2.putText(annotated, left_text, left_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        
        # Right eye label
        right_label_pos = tuple(right_center - [0, 25])
        cv2.rectangle(annotated,
                    (right_label_pos[0] - 5, right_label_pos[1] - 20),
                    (right_label_pos[0] + 100, right_label_pos[1] + 5),
                    (0, 0, 0), -1)
        cv2.putText(annotated, right_text, right_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        
        # Draw drowsiness alert with pulsing effect
        if is_drowsy:
            # Large warning at top
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), (0, 0, 200), -1)
            cv2.putText(annotated, "DROWSINESS ALERT!", (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        
        # Draw status bar at bottom with better styling
        status_y = annotated.shape[0] - 40
        cv2.rectangle(annotated, (0, status_y - 10), (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
        
        status_text = f"Closed Frames: {self.CLOSED_FRAMES}/{self.DROWSINESS_THRESHOLD}"
        cv2.putText(annotated, status_text, (50, status_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add progress bar
        bar_width = 300
        bar_x = annotated.shape[1] - bar_width - 50
        bar_y = status_y + 5
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100, 100, 100), 2)
        
        # Fill progress
        fill_width = int((self.CLOSED_FRAMES / self.DROWSINESS_THRESHOLD) * bar_width)
        fill_width = min(fill_width, bar_width)
        if fill_width > 0:
            bar_color = (0, 0, 255) if is_drowsy else (0, 165, 255)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_width, bar_y + 20), bar_color, -1)
        
        return annotated
    
    # Draw no face detected warning
    def _draw_no_face_warning(self, frame):
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        
        # Draw warning text
        cv2.putText(annotated, "No Face Detected", (w // 2 - 180, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(annotated, "Please position your face in frame", (w // 2 - 220, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated
    
    # Reset all counters and history buffers
    def reset(self):
        self.CLOSED_FRAMES = 0
        self.eye_state_history = {'left': [], 'right': []}