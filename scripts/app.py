import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# 1. Page Configuration
st.set_page_config(page_title="Structural Defect Detector", layout="wide")
st.title("Drone Diagnostics: Structure & Defect Segmentation")
st.markdown("Upload a drone image to detect structures (towers, bridges) and surface decay (corrosion, cracks).")

# 2. Sidebar Settings
st.sidebar.header("Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# 3. Load the Model (with Caching)
# st.cache_resource ensures Hugging Face only loads the AI once, saving RAM.
@st.cache_resource
def load_model():
    # Dynamically find the absolute path to the directory this script is in (scripts folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate UP one level to the root directory, then into the models folder
    root_dir = os.path.dirname(current_dir)
    weights_path = os.path.join(root_dir, 'models', 'best.pt')
    
    # NEW: Let's prove mathematically if the file is there or not
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Python looked exactly here: {weights_path} but it was empty.")
        
    return YOLO(weights_path)

# Safely attempt to load the model and expose the TRUE error
try:
    model = load_model()
except Exception as e:
    st.error(f"🚨 Critical Model Error: {e}")
    st.info("Screenshot this error so we can debug exactly what went wrong!")
    st.stop()

# 4. Image Upload Widget
uploaded_file = st.file_uploader("Choose a drone image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    max_size = 1280
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Create two columns for a side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.header("Segmentation Results")
        
        # Show a loading spinner while the cloud CPU runs the inference
        with st.spinner("Analyzing structural integrity..."):
            
            # Run the YOLO inference on the uploaded image
            results = model.predict(source=image, conf=conf_threshold)
            
            # --- THE INFERENCE TRICK (Merging Classes) ---
            # We intercept the model's internal dictionary before drawing the image.
            # If it predicted "alligator crack" or "concrete crack", it will display just "crack".
            names_dict = results[0].names
            for key, value in names_dict.items():
                if "crack" in value.lower():
                    names_dict[key] = "crack"
                elif "corrosion" in value.lower():
                    names_dict[key] = "corrosion"
            
            # Plot the masks and bounding boxes using the updated class names
            res_plotted = results[0].plot()
            
            # Display the final analyzed image 
            # (channels="BGR" is required because OpenCV outputs in Blue-Green-Red format)
            st.image(res_plotted, channels="BGR", use_container_width=True)