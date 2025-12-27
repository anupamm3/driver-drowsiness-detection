class DrowsinessDetector:
    def __init__(self, model_path):
        import cv2
        import mediapipe as mp
        from tensorflow.keras.models import load_model

        self.model = load_model(model_path)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.CLOSED_FRAMES = 0
        self.YAWN_FRAMES = 0

    def process_frame(self, frame):
        import numpy as np

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Extract eye landmarks
            left_eye = self.extract_eye(landmarks, left=True)
            right_eye = self.extract_eye(landmarks, left=False)

            # Process eyes
            left_eye_state = self.process_eye(left_eye)
            right_eye_state = self.process_eye(right_eye)

            # Check for drowsiness
            if left_eye_state == 'Closed' and right_eye_state == 'Closed':
                self.CLOSED_FRAMES += 1
            else:
                self.CLOSED_FRAMES = 0

            # Check for yawning
            mouth_aspect_ratio = self.calculate_mar(landmarks)
            if mouth_aspect_ratio > 0.5:
                self.YAWN_FRAMES += 1
            else:
                self.YAWN_FRAMES = 0

            return {
                'left_eye': left_eye_state,
                'score': self.CLOSED_FRAMES,
                'yawn': self.YAWN_FRAMES > 0,
                'annotated_frame': frame
            }
        else:
            return {
                'left_eye': 'No Face Detected',
                'score': 0,
                'yawn': False,
                'annotated_frame': frame
            }

    def extract_eye(self, landmarks, left=True):
        # Define eye landmarks indices
        if left:
            indices = [33, 133, 153, 144, 163, 7, 163, 144, 153, 133]
        else:
            indices = [362, 263, 283, 274, 293, 1, 293, 274, 283, 263]

        eye_landmarks = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in indices]
        return eye_landmarks

    def process_eye(self, eye_landmarks):
        import cv2

        # Crop, resize, and predict
        eye_image = self.prepare_eye_image(eye_landmarks)
        prediction = self.model.predict(eye_image)
        return 'Closed' if prediction[0][1] > 0.9 else 'Open'

    def prepare_eye_image(self, eye_landmarks):
        # Convert landmarks to image
        eye_image = np.zeros((64, 64, 1), dtype=np.uint8)
        # Logic to fill eye_image based on eye_landmarks
        return eye_image.reshape(1, 64, 64, 1)

    def calculate_mar(self, landmarks):
        # Calculate Mouth Aspect Ratio (MAR)
        upper_lip = (landmarks.landmark[13].y + landmarks.landmark[14].y) / 2
        lower_lip = (landmarks.landmark[19].y + landmarks.landmark[20].y) / 2
        return (upper_lip - lower_lip) / (landmarks.landmark[61].x - landmarks.landmark[291].x)