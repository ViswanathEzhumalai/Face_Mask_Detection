# Face_Mask_Detection

This project uses deep learning to detect if a person is wearing a face mask. It includes a Google Colab notebook for training and a Python script for live camera detection.



Features:

Train a face mask detection model using Google Colab.

Real-time detection using a webcam.



Instructions:

Clone the repository: bash git clone https://github.com/your-username/FaceMaskDetection.git



Training:

Open train_face_mask_detector.ipynb in Google Colab.

Follow the instructions to train the model and save it to your Google Drive.



Live Detection:

Place the trained model (face_mask_detector.h5) in the same directory as live_camera_mask_detection.py.

Run the script: bash python live_camera_mask_detection.py

Use 'q' to quit the live detection.



Dataset Structure:

The dataset should be organized as:

dataset/
  train/
    WithMask/
    WithoutMask/
  validation/
    WithMask/
    WithoutMask/


    
License:
MIT License.

