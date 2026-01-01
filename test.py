## Test the detector on video

# import cv2
# import os
# from src.detector import DrowsinessDetector

# def test_video_detailed(video_path):
#     # Check if file exists
#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found at: {video_path}")
#         print(f"Please check the file path and try again.")
#         return
    
#     detector = DrowsinessDetector(model_path='models/drowsiness_cnn.h5')
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video file: {video_path}")
#         return
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print("\n" + "="*80)
#     print(f"VIDEO ANALYSIS: {video_path}")
#     print("="*80)
#     print(f"Total Frames: {total_frames}")
#     print(f"FPS: {fps:.2f}")
#     print(f"Duration: {total_frames/fps:.2f} seconds")
#     print("="*80)
#     print(f"\n{'Frame':<8} {'Time(s)':<10} {'Left':<25} {'Right':<25} {'Both Closed?'}")
#     print("-"*80)
    
#     frame_num = 0
#     closed_count = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_num += 1
#         result = detector.process_frame(frame)
        
#         both_closed = (result['left_eye_state'] == 'Closed' and 
#                       result['right_eye_state'] == 'Closed')
        
#         if both_closed:
#             closed_count += 1
        
#         # Print every 15 frames (~0.5 seconds)
#         if frame_num % 15 == 0 or both_closed:
#             time_sec = frame_num / fps
#             left_info = f"{result['left_eye_state']} ({result['left_confidence']:.2f})"
#             right_info = f"{result['right_eye_state']} ({result['right_confidence']:.2f})"
#             marker = "✓" if both_closed else " "
            
#             print(f"{frame_num:<8} {time_sec:<10.2f} {left_info:<25} {right_info:<25} {marker}")
        
#         # Show frame (optional - comment out if you don't want the window)
#         cv2.imshow('Analysis', result['annotated_frame'])
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nAnalysis interrupted by user (pressed 'q')")
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
    
#     print("="*80)
#     print(f"\nRESULTS:")
#     print(f"   Total Frames Analyzed: {frame_num}")
#     print(f"   Frames with Both Eyes Closed: {closed_count}")
#     print(f"   Percentage Closed: {(closed_count/frame_num)*100:.1f}%")
#     print(f"   Duration Closed: {(closed_count/fps):.2f} seconds")
#     print(f"\n   Expected (5s closed / 8s video): ~62.5% closed")
#     print(f"   Actual: {(closed_count/frame_num)*100:.1f}% closed")
    
#     if (closed_count/frame_num)*100 > 50:
#         print(f"\n    PASS: Detection working correctly!")
#     else:
#         print(f"\n    FAIL: Still detecting mostly open")
#         print(f"   → Try: python check_model_classes.py")
#         print(f"   → Your model's class indices might be reversed")
    
#     print("="*80)

# if __name__ == "__main__":
#     # Default video path pointing to assets/vid1
#     default_video = "assets/videos/vid1.mp4"
    
#     # Check common video extensions if .mp4 doesn't exist
#     if not os.path.exists(default_video):
#         for ext in ['.avi', '.mov', '.mkv', '.mp4']:
#             test_path = f"assets/videos/vid1{ext}"
#             if os.path.exists(test_path):
#                 default_video = test_path
#                 break
    
#     print("\n" + "="*80)
#     print(" DRIVER DROWSINESS DETECTION - VIDEO ANALYSIS")
#     print("="*80)
#     print(f"\nDefault video: {default_video}")
    
#     video_path = input("\nEnter video path (or press Enter to use default): ").strip()
    
#     if not video_path:
#         video_path = default_video
#         print(f" Using default video: {video_path}")
    
#     test_video_detailed(video_path)







# Test with EAR debug output

import cv2
from src.detector import DrowsinessDetector

video_path = "assets/videos/vid1.mp4"
detector = DrowsinessDetector(model_path='models/drowsiness_cnn.h5')
cap = cv2.VideoCapture(video_path)

frame_num = 0
closed_count = 0

print("\n" + "="*100)
print(f"{'Frame':<8} {'Time(s)':<10} {'Left EAR':<12} {'Right EAR':<12} {'Left State':<25} {'Right State':<25} {'Both Closed?'}")
print("-"*100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num += 1
    result = detector.process_frame(frame)
    
    both_closed = (result['left_eye_state'] == 'Closed' and result['right_eye_state'] == 'Closed')
    
    if both_closed:
        closed_count += 1
    
    # Print every 15 frames
    if frame_num % 15 == 0:
        time_sec = frame_num / 25.0
        left_info = f"{result['left_eye_state']} ({result['left_confidence']:.2f})"
        right_info = f"{result['right_eye_state']} ({result['right_confidence']:.2f})"
        marker = "✓" if both_closed else " "
        
        print(f"{frame_num:<8} {time_sec:<10.2f} {'N/A':<12} {'N/A':<12} {left_info:<25} {right_info:<25} {marker}")
    
    cv2.imshow('Test', result['annotated_frame'])
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("="*100)
print(f"\n RESULTS:")
print(f"   Total Frames: {frame_num}")
print(f"   Frames with Both Eyes Closed: {closed_count}")
print(f"   Percentage Closed: {(closed_count/frame_num)*100:.1f}%")
print(f"   Expected: ~62.5% (5s closed / 8s video)")
print(f"   {' PASS' if (closed_count/frame_num)*100 > 50 else ' FAIL'}")
print("="*100)