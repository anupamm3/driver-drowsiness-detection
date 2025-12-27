import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


class DrowsinessDetector:
    """
    DrowsinessDetector class for real-time drowsiness detection.
    Uses MediaPipe Face Mesh for facial landmark detection and a CNN for eye state classification.
    """
    
    def __init__(self, model_path):
        """
        Initialize the DrowsinessDetector.
        
        Args:
            model_path (str): Path to the trained CNN model (.h5 file)
        """
        # Load the trained CNN model for eye state classification
        self.model = load_model(model_path)
        
        # Initialize MediaPipe Face Mesh with refined landmarks for better eye detection
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Counter for consecutive closed eye frames (drowsiness indicator)
        self.CLOSED_FRAMES = 0
        
        # Counter for consecutive yawn frames
        self.YAWN_FRAMES = 0

    def process_frame(self, frame):
        """
        Process a single video frame to detect drowsiness and yawning.
        
        Args:
            frame: Input video frame (BGR format from OpenCV)
            
        Returns:
            dict: Dictionary containing detection results with keys:
                - 'left_eye': State of left eye ('Open', 'Closed', or 'No Face Detected')
                - 'score': Drowsiness score (number of consecutive closed frames)
                - 'yawn': Boolean indicating if yawning is detected
                - 'annotated_frame': The input frame (can be modified for visualization)
        """
        # Convert frame from BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        results = self.mp_face_mesh.process(rgb_frame)

        # Check if any face was detected in the frame
        if results.multi_face_landmarks:
            # Get the first detected face's landmarks
            landmarks = results.multi_face_landmarks[0]

            # Extract eye region landmarks for both eyes
            left_eye = self.extract_eye(landmarks, left=True)
            right_eye = self.extract_eye(landmarks, left=False)

            # Process both eyes through the CNN to determine their state
            left_eye_state = self.process_eye(left_eye, frame)
            right_eye_state = self.process_eye(right_eye, frame)

            # Drowsiness detection: Both eyes must be closed
            if left_eye_state == 'Closed' and right_eye_state == 'Closed':
                self.CLOSED_FRAMES += 1
            else:
                # Reset counter if eyes are open
                self.CLOSED_FRAMES = 0

            # Yawn detection using Mouth Aspect Ratio (MAR)
            mouth_aspect_ratio = self.calculate_mar(landmarks)
            if mouth_aspect_ratio > 0.5:
                self.YAWN_FRAMES += 1
            else:
                self.YAWN_FRAMES = 0

            # Return detection results
            return {
                'left_eye': left_eye_state,
                'score': self.CLOSED_FRAMES,
                'yawn': self.YAWN_FRAMES > 0,
                'annotated_frame': frame
            }
        else:
            # No face detected - reset all counters
            return {
                'left_eye': 'No Face Detected',
                'score': 0,
                'yawn': False,
                'annotated_frame': frame
            }

    def extract_eye(self, landmarks, left=True):
        """
        Extract eye landmark coordinates from the face mesh.
        
        Args:
            landmarks: MediaPipe face mesh landmarks
            left (bool): True for left eye, False for right eye
            
        Returns:
            list: List of (x, y) tuples representing eye landmark coordinates
        """
        # MediaPipe Face Mesh landmark indices for eyes
        # Left eye indices: Key points around the left eye contour
        # Right eye indices: Key points around the right eye contour
        if left:
            indices = [33, 133, 153, 144, 163, 7]
        else:
            indices = [362, 263, 283, 274, 293, 1]

        # Extract normalized (x, y) coordinates for each landmark
        eye_landmarks = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in indices]
        return eye_landmarks

    def process_eye(self, eye_landmarks, frame):
        """
        Process eye landmarks through the CNN to classify eye state.
        
        Args:
            eye_landmarks: List of eye landmark coordinates
            frame: Original video frame for reference
            
        Returns:
            str: 'Closed' if eye is closed (prediction > 0.9), else 'Open'
        """
        # Prepare the eye image for CNN input
        eye_image = self.prepare_eye_image(eye_landmarks, frame)
        
        # Run prediction through the trained CNN model
        prediction = self.model.predict(eye_image, verbose=0)
        
        # Classification: Index 0 = Open, Index 1 = Closed
        # Threshold: 0.9 confidence required to classify as "Closed"
        return 'Closed' if prediction[0][1] > 0.9 else 'Open'

    def prepare_eye_image(self, eye_landmarks, frame):
        """
        Prepare eye region image for CNN input.
        Crops, converts to grayscale, and resizes to (64, 64).
        
        Args:
            eye_landmarks: List of eye landmark coordinates (normalized)
            frame: Original video frame
            
        Returns:
            numpy.ndarray: Preprocessed eye image ready for CNN (shape: 1, 64, 64, 1)
        """
        h, w, _ = frame.shape
        
        # Convert normalized coordinates to pixel coordinates
        points = np.array([(int(x * w), int(y * h)) for x, y in eye_landmarks])
        
        # Calculate bounding box around the eye
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add padding to the bounding box (10 pixels on each side)
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop the eye region from the frame
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        # Handle edge case: empty crop
        if eye_region.size == 0:
            eye_region = np.zeros((64, 64, 1), dtype=np.uint8)
        else:
            # Convert to grayscale (required by the CNN)
            eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to 64x64 (CNN input size)
            eye_region = cv2.resize(eye_region, (64, 64))
            
            # Add channel dimension for grayscale
            eye_region = np.expand_dims(eye_region, axis=-1)
        
        # Normalize pixel values to [0, 1] range
        eye_region = eye_region.astype('float32') / 255.0
        
        # Add batch dimension (1, 64, 64, 1)
        return eye_region.reshape(1, 64, 64, 1)

    def calculate_mar(self, landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) to detect yawning.
        
        Formula: MAR = vertical_distance / horizontal_distance
        Higher MAR indicates mouth is open (yawning).
        
        Args:
            landmarks: MediaPipe face mesh landmarks
            
        Returns:
            float: Mouth Aspect Ratio value
        """
        # Upper lip landmarks (top of mouth)
        upper_lip = (landmarks.landmark[13].y + landmarks.landmark[14].y) / 2
        
        # Lower lip landmarks (bottom of mouth)
        lower_lip = (landmarks.landmark[19].y + landmarks.landmark[20].y) / 2
        
        # Mouth corners (left and right)
        mouth_width = abs(landmarks.landmark[61].x - landmarks.landmark[291].x)
        
        # Calculate vertical distance (mouth opening)
        vertical_distance = abs(lower_lip - upper_lip)
        
        # Avoid division by zero
        if mouth_width == 0:
            return 0
        
        # MAR = vertical / horizontal
        mar = vertical_distance / mouth_width
        
        return mar