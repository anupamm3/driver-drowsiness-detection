import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from src.detector import DrowsinessDetector
import time

# Page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
        margin: 1rem 0;
    }
    .safe-box {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00ff00;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_frames': 0,
        'drowsy_frames': 0,
        'alert_count': 0
    }

# Header
st.markdown('<h1 class="main-header">🚗 Driver Drowsiness Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Settings
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=100)
    st.title("⚙️ Settings")
    
    st.markdown("### Model Configuration")
    model_path = st.text_input("Model Path", value="models/drowsiness_cnn.h5", disabled=True)
    
    st.markdown("### Detection Parameters")
    drowsiness_threshold = st.slider(
        "Drowsiness Threshold (frames)", 
        min_value=5, 
        max_value=50, 
        value=15,
        help="Number of consecutive closed-eye frames to trigger alert"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5,
        step=0.05,
        help="Minimum confidence to classify eye as closed"
    )
    
    st.markdown("### Video Settings")
    process_every_n_frames = st.slider(
        "Process Every N Frames", 
        min_value=1, 
        max_value=10, 
        value=2,
        help="Skip frames for faster processing (1 = process all frames)"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **Drowsiness Detection System**
    
    Uses CNN to detect closed eyes and alert when driver shows signs of drowsiness.
    
    **Features:**
    - Real-time eye state detection
    - Configurable sensitivity
    - Video file support
    - Statistical analysis
    """)

# Initialize detector
@st.cache_resource
def load_detector(model_path):
    """Load the drowsiness detector (cached)."""
    try:
        detector = DrowsinessDetector(model_path=model_path)
        detector.DROWSINESS_THRESHOLD = drowsiness_threshold
        detector.EYE_CLOSED_THRESHOLD = confidence_threshold
        return detector
    except Exception as e:
        st.error(f"❌ Error loading detector: {e}")
        return None

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">📹 Video Input</h2>', unsafe_allow_html=True)
    
    # Video source selection
    video_source = st.radio(
        "Select Video Source:",
        options=["Upload Video File", "Use Webcam (Real-time)"],
        horizontal=True
    )
    
    if video_source == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload a video file (MP4, AVI, MOV)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Start processing button
            if st.button("🚀 Start Detection", type="primary", use_container_width=True):
                st.session_state.processing = True
        else:
            video_path = None
    else:
        st.info("💡 Click 'Start Webcam' to begin real-time detection")
        if st.button("📷 Start Webcam", type="primary", use_container_width=True):
            st.session_state.processing = True
            video_path = 0  # Webcam
        else:
            video_path = None

with col2:
    st.markdown('<h2 class="sub-header">📊 Statistics</h2>', unsafe_allow_html=True)
    
    # Metrics display
    metric_placeholder = st.empty()
    
    # Status indicator
    status_placeholder = st.empty()

# Video processing area
video_placeholder = st.empty()
progress_placeholder = st.empty()

# Process video
if st.session_state.processing:
    # Load detector
    detector = load_detector(model_path)
    
    if detector is None:
        st.error("❌ Failed to load detector. Please check model path.")
        st.stop()
    
    # Update detector parameters
    detector.DROWSINESS_THRESHOLD = drowsiness_threshold
    detector.EYE_CLOSED_THRESHOLD = confidence_threshold
    
    # Open video capture
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Error: Could not open video source")
            st.stop()
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if video_path == 0:  # Webcam
            total_frames = 0  # Unknown for webcam
            st.info("🔴 Live webcam feed - Press 'Stop' to end")
        
        # Add stop button
        stop_button = st.button("🛑 Stop Detection", type="secondary")
        
        frame_count = 0
        drowsy_count = 0
        alert_triggered = False
        
        # Processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret or stop_button:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % process_every_n_frames != 0:
                continue
            
            # Process frame
            result = detector.process_frame(frame)
            
            # Update statistics
            if result['drowsy']:
                drowsy_count += 1
                if not alert_triggered:
                    st.session_state.stats['alert_count'] += 1
                    alert_triggered = True
            else:
                alert_triggered = False
            
            st.session_state.stats['total_frames'] = frame_count
            st.session_state.stats['drowsy_frames'] = drowsy_count
            
            # Display annotated frame
            video_placeholder.image(
                cv2.cvtColor(result['annotated_frame'], cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_column_width=True
            )
            
            # Update metrics
            with metric_placeholder.container():
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Frames", frame_count)
                with m2:
                    st.metric("Drowsy Frames", drowsy_count)
                with m3:
                    drowsy_percentage = (drowsy_count / frame_count * 100) if frame_count > 0 else 0
                    st.metric("Drowsy %", f"{drowsy_percentage:.1f}%")
            
            # Status indicator
            with status_placeholder.container():
                if result['drowsy']:
                    st.markdown("""
                        <div class="alert-box">
                            <h3>⚠️ DROWSINESS ALERT!</h3>
                            <p>Driver appears drowsy. Eyes closed for {0} frames.</p>
                        </div>
                    """.format(result['score']), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="safe-box">
                            <h3>✅ Driver Alert</h3>
                            <p>Left Eye: {0} | Right Eye: {1}</p>
                        </div>
                    """.format(result['left_eye_state'], result['right_eye_state']), 
                    unsafe_allow_html=True)
            
            # Update progress bar (for video files)
            if total_frames > 0:
                progress = frame_count / total_frames
                progress_placeholder.progress(progress, text=f"Processing: {frame_count}/{total_frames} frames")
            
            # Small delay for webcam
            if video_path == 0:
                time.sleep(0.03)  # ~30 FPS
        
        # Cleanup
        cap.release()
        
        # Remove temporary file
        if video_path != 0:
            try:
                os.unlink(video_path)
            except:
                pass
        
        # Final summary
        st.success("✅ Processing complete!")
        
        st.markdown("### 📈 Final Report")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames Processed", st.session_state.stats['total_frames'])
        with col2:
            st.metric("Drowsy Frames Detected", st.session_state.stats['drowsy_frames'])
        with col3:
            total = st.session_state.stats['total_frames']
            drowsy = st.session_state.stats['drowsy_frames']
            percentage = (drowsy / total * 100) if total > 0 else 0
            st.metric("Drowsiness Percentage", f"{percentage:.2f}%")
        with col4:
            st.metric("Alert Count", st.session_state.stats['alert_count'])
        
        st.session_state.processing = False
        
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
        st.session_state.processing = False

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Driver Drowsiness Detection System | Built with Streamlit & TensorFlow</p>
        <p>⚠️ For educational purposes only. Not a substitute for proper rest.</p>
    </div>
""", unsafe_allow_html=True)