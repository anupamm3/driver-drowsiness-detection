import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

class DrowsinessDetector:
    """
    Drowsiness detection using MediaPipe face mesh and trained CNN.
    """
    
    def __init__(self, model_path):
        # Load CNN model
        try:
            self.model = load_model(model_path)
            print(f"✅ CNN Model loaded successfully from: {model_path}")  # ADD THIS
        except Exception as e:
            raise Exception(f"❌ Error loading CNN model: {e}")
        
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
            print("✅ MediaPipe Face Mesh initialized successfully")  # ADD THIS
        except Exception as e:
            raise Exception(f"❌ Error initializing MediaPipe: {e}")
        
        # Eye landmark indices (MediaPipe uses 468 points)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]  # 6 key points (similar to dlib)
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Alternative: Use all eye points for better accuracy
        # self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Drowsiness tracking
        self.CLOSED_FRAMES = 0
        self.DROWSINESS_THRESHOLD = 15
        self.EYE_CLOSED_THRESHOLD = 0.5
    
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
                'face_detected': False,
                'annotated_frame': self._draw_no_face_warning(frame)
            }
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract eye landmarks
        left_eye = self._get_landmarks(face_landmarks, self.LEFT_EYE, w, h)
        right_eye = self._get_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
        
        # Predict eye states
        left_eye_state, left_confidence = self._predict_eye_state(frame, left_eye)
        right_eye_state, right_confidence = self._predict_eye_state(frame, right_eye)
        
        # Update closed frames counter
        if left_eye_state == 'Closed' and right_eye_state == 'Closed':
            self.CLOSED_FRAMES += 1
        else:
            self.CLOSED_FRAMES = max(0, self.CLOSED_FRAMES - 1)
        
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
    
    def _get_landmarks(self, face_landmarks, indices, width, height):
        """Convert MediaPipe landmarks to pixel coordinates."""
        landmarks = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y])
        return np.array(landmarks, dtype=np.int32)
    
    def _predict_eye_state(self, frame, eye_landmarks):
        """Predict eye state using CNN (same as before)."""
        eye_image = self._extract_eye_region(frame, eye_landmarks)
        
        if eye_image is None:
            return 'Unknown', 0.0
        
        prediction = self.model.predict(eye_image, verbose=0)
        closed_confidence = prediction[0][0]
        open_confidence = prediction[0][1]
        
        if open_confidence > self.EYE_CLOSED_THRESHOLD:
            return 'Open', open_confidence
        else:
            return 'Closed', closed_confidence
    
    def _extract_eye_region(self, frame, eye_landmarks):
        """Extract eye region (same as before)."""
        x_coords = eye_landmarks[:, 0]
        y_coords = eye_landmarks[:, 1]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        padding = 10
        h, w = frame.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0 or eye_region.shape[0] < 10 or eye_region.shape[1] < 10:
            return None
        
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_resized = cv2.resize(eye_gray, (64, 64))
        eye_normalized = eye_resized.astype('float32') / 255.0
        eye_final = eye_normalized.reshape(1, 64, 64, 1)
        
        return eye_final
    
    def _annotate_frame(self, frame, left_eye, right_eye,
                       left_state, right_state, left_conf, right_conf, is_drowsy):
        """Draw annotations (similar to before)."""
        annotated = frame.copy()
        
        # Draw eye contours
        left_color = (0, 0, 255) if left_state == 'Closed' else (0, 255, 0)
        right_color = (0, 0, 255) if right_state == 'Closed' else (0, 255, 0)
        
        cv2.polylines(annotated, [left_eye], True, left_color, 2)
        cv2.polylines(annotated, [right_eye], True, right_color, 2)
        
        # Draw labels
        left_center = left_eye.mean(axis=0).astype(int)
        right_center = right_eye.mean(axis=0).astype(int)
        
        cv2.putText(annotated, f"{left_state} ({left_conf:.2f})",
                   tuple(left_center - [0, 20]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        
        cv2.putText(annotated, f"{right_state} ({right_conf:.2f})",
                   tuple(right_center - [0, 20]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        
        # Drowsiness warning
        if is_drowsy:
            cv2.putText(annotated, "DROWSINESS ALERT!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.putText(annotated, f"Closed Frames: {self.CLOSED_FRAMES}/{self.DROWSINESS_THRESHOLD}",
                   (50, annotated.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def _draw_no_face_warning(self, frame):
        """Draw no face warning."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        cv2.putText(annotated, "No Face Detected",
                   (w // 2 - 150, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return annotated
    
    def reset(self):
        """Reset counters."""
        self.CLOSED_FRAMES = 0