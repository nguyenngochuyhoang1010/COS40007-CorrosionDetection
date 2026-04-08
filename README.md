🏗️ Drone Diagnostics: Structural Defect Segmenter

A computer vision pipeline utilizing a custom-trained YOLOv26 Nano model to perform instance segmentation on concrete structures. This tool is designed to process drone imagery to simultaneously identify structural boundaries (via polygon masks) and isolate surface-level decay such as cracks (via bounding boxes).

🚀 Try the Live Web App on Hugging Face Spaces!
https://huggingface.co/spaces/venenacoenubia/structural-defect-detector

📂 Repository Structure

concrete-defect-segmentation/
│
├── models/                     
│   └── best.pt                 # Custom-trained YOLOv26 Nano weights
│
├── scripts/                    
│   └── app.py                  # Streamlit Web User Interface
│
├── .gitignore                  # Ignores large datasets and python cache
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


🚀 Getting Started (Local Deployment)

To run this AI demonstrator on your local machine, follow these steps:

1. Clone the Repository

git clone [https://github.com/nguyenngochuyhoang1010/COS40007-CorrosionDetection](https://github.com/nguyenngochuyhoang1010/COS40007-CorrosionDetection)
cd COS40007-CorrosionDetection


2. Install Dependencies

It is highly recommended to use a Python virtual environment. Install the required packages using:

pip install -r requirements.txt


3. Run the Web Interface

Launch the Streamlit demonstrator by running the app.py file located in the scripts folder:

streamlit run scripts/app.py


The app will automatically open in your default web browser at http://localhost:8501.

🧠 Model Performance

Performance Baseline (50 Epochs): * ~70% mAP@0.5 for Bounding Boxes (Localization)

~65% mAP@0.5 for Segmentation Masks (Pixel mapping)

Hardware Note: The Nano architecture was deliberately selected to allow for highly efficient, 30+ FPS real-time inference on edge devices (such as drone onboard computers) without suffering from CUDA Out-of-Memory limits.
👥 Team Members

Nguyen Ngoc Huy Hoang
Nguyen Xuan Duy Thai
Nguyen Dang Vinh
