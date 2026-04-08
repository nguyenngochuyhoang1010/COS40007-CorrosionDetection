import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import tempfile

# 1. Page Configuration
st.set_page_config(page_title="Drone Diagnostics", page_icon="🏗️", layout="wide")
st.title("🏗️ Drone Diagnostics: Structure & Defect Segmentation")

# 2. Sidebar Settings
st.sidebar.header("Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# 3. Input Mode Selector
st.sidebar.markdown("---")
input_mode = st.sidebar.radio("Select Input Mode:", ["Image Upload", "Video Upload", "Live Camera"])

# 4. Load the Model (with Caching)
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    weights_path = os.path.join(root_dir, 'models', 'best.pt')
    
    if not os.path.exists(weights_path):
        # Fallback for Hugging Face flat directory structure if models/ isn't used
        weights_path = os.path.join(current_dir, 'best.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
    return YOLO(weights_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"🚨 Critical Model Error: {e}")
    st.stop()


# ==========================================
# MODE 1: IMAGE UPLOAD
# ==========================================
if input_mode == "Image Upload":
    st.write("Upload a drone image to detect structures and surface decay.")
    
    # NEW: Add a highly visible pro-tip so users know about the secret paste feature!
    st.info("💡 **Pro Tip:** You can take a screenshot, click anywhere on the **blank background** of this app (outside the grey box), and press **Ctrl+V** (or **Cmd+V** on Mac) to paste it directly!")
    
    # Update the label to make it obvious
    uploaded_file = st.file_uploader("Drag & drop, browse, or press Ctrl+V anywhere on the page...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Safety Resize to prevent cloud RAM crashes
        max_size = 640
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            with st.spinner("Analyzing structural integrity..."):
                results = model.predict(source=image, conf=conf_threshold)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Segmentation Results", channels="BGR", use_container_width=True)


# ==========================================
# MODE 2: VIDEO UPLOAD
# ==========================================
elif input_mode == "Video Upload":
    st.write("Upload drone video footage for frame-by-frame defect analysis.")
    st.warning("⚠️ Note: Video processing on cloud CPUs is slow. Short clips (under 10 seconds) are recommended.")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        if st.button("Start Video Analysis"):
            with st.spinner("Processing video... this may take a while."):
                # Save uploaded video to a temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                
                # Open the video with OpenCV
                cap = cv2.VideoCapture(tfile.name)
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # --- NEW: SAFETY RESIZE FOR VIDEO ---
                # Shrink video resolution to prevent cloud crashes and speed up inference
                max_vid_size = 640
                if max(orig_width, orig_height) > max_vid_size:
                    scale = max_vid_size / max(orig_width, orig_height)
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)
                else:
                    new_width = orig_width
                    new_height = orig_height

                # Create an output temporary file
                out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                out = cv2.VideoWriter(out_file.name, fourcc, fps, (new_width, new_height))

                progress_bar = st.progress(0)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize the frame BEFORE giving it to the AI
                    frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Run YOLO inference on the frame
                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    res_frame = results[0].plot()
                    
                    # Write to the output video
                    out.write(res_frame)
                    
                    # Update progress bar
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                # Release resources
                cap.release()
                out.release()
                
                st.success("Video processing complete!")
                st.video(out_file.name)


# ==========================================
# MODE 3: LIVE CAMERA
# ==========================================
elif input_mode == "Live Camera":
    st.write("Use your device's camera to capture structures in the field.")
    # Streamlit's built-in camera widget
    camera_photo = st.camera_input("Take a photo to analyze")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        
        with st.spinner("Analyzing structural integrity..."):
            results = model.predict(source=image, conf=conf_threshold)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Segmentation Results", channels="BGR", use_container_width=True)
            
            # Display the final analyzed image 
            # (channels="BGR" is required because OpenCV outputs in Blue-Green-Red format)
            st.image(res_plotted, channels="BGR", use_container_width=True)
