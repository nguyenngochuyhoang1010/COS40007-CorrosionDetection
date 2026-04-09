import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import tempfile

# 1. Page Configuration
st.set_page_config(page_title="Drone Diagnostics", page_icon="🏗️", layout="wide")
st.title("🏗️ Drone Diagnostics: Structure & Defect Segmentation")

# 2. Sidebar Settings & Session Management
st.sidebar.header("Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# --- NEW: VISUAL TOGGLES ---
st.sidebar.markdown("---")
st.sidebar.header("Visual Settings")
show_boxes = st.sidebar.toggle("Show Bounding Boxes", value=True)
show_labels = st.sidebar.toggle("Show Class Labels", value=True)
# ---------------------------

if st.sidebar.button("🧹 Clear Session / Reset App"):
    st.cache_resource.clear()
    st.rerun()

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
    
    st.info("""
    💡 **How to Paste (Chrome/Edge):**
    1. Click the **Lock Icon** 🔒 in your browser's address bar and set **Clipboard** to **'Allow'**.
    2. Click once on the **sidebar** to focus the window.
    3. Press **Ctrl+V**. 
    
    *If it still fails, your browser may be blocking clipboard access for security. Please use **Drag & Drop** instead.*
    """)
    
    uploaded_file = st.file_uploader("Drag & drop, browse, or press Ctrl+V anywhere...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Match training resolution: 640px
            max_size = 640
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original (Optimized to 640px)", use_container_width=True)

            with col2:
                with st.spinner("Analyzing structural integrity..."):
                    results = model.predict(source=image, conf=conf_threshold, imgsz=640)
                    
                    # --- NEW: APPLY TOGGLES TO THE PLOT ---
                    res_plotted = results[0].plot(boxes=show_boxes, labels=show_labels)
                    # --------------------------------------
                    
                    st.image(res_plotted, caption="Segmentation Results", channels="BGR", use_container_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}. Try clicking 'Clear Session' in the sidebar.")


# ==========================================
# MODE 2: VIDEO UPLOAD
# ==========================================
elif input_mode == "Video Upload":
    st.write("Upload drone video footage for frame-by-frame defect analysis.")
    st.warning("⚠️ Note: Video processing on cloud CPUs is slow. Short clips (under 10 seconds) are recommended.")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        if st.button("Start Video Analysis"):
            with st.spinner("Processing video... matching to training resolution (640px)."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                
                cap = cv2.VideoCapture(tfile.name)
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                max_vid_size = 640
                if max(orig_width, orig_height) > max_vid_size:
                    scale = max_vid_size / max(orig_width, orig_height)
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)
                else:
                    new_width = orig_width
                    new_height = orig_height

                out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                out = cv2.VideoWriter(out_file.name, fourcc, fps, (new_width, new_height))

                progress_bar = st.progress(0)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.resize(frame, (new_width, new_height))
                    results = model.predict(frame, conf=conf_threshold, imgsz=640, verbose=False)
                    
                    # --- NEW: APPLY TOGGLES TO THE PLOT ---
                    res_frame = results[0].plot(boxes=show_boxes, labels=show_labels)
                    # --------------------------------------
                    
                    out.write(res_frame)
                    
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                out.release()
                st.success("Video processing complete!")
                st.video(out_file.name)


# ==========================================
# MODE 3: LIVE CAMERA
# ==========================================
elif input_mode == "Live Camera":
    st.write("Use your device's camera to capture structures in the field.")
    camera_photo = st.camera_input("Take a photo to analyze")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        max_size = 640
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        with st.spinner("Analyzing structural integrity..."):
            results = model.predict(source=image, conf=conf_threshold, imgsz=640)
            
            # --- NEW: APPLY TOGGLES TO THE PLOT ---
            res_plotted = results[0].plot(boxes=show_boxes, labels=show_labels)
            # --------------------------------------
            
            st.image(res_plotted, caption="Segmentation Results", channels="BGR", use_container_width=True)
