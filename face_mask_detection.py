
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("face_mask_detector.h5")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Load a pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define image size for model input
IMG_SIZE = (100, 100)

# Function to predict mask status
def detect_mask(face):
    face = cv2.resize(face, IMG_SIZE)
    face = np.expand_dims(face, axis=0) / 255.0  # Normalize
    prediction = model.predict(face)[0][0]
    return "With Mask" if prediction < 0.5 else "Without Mask"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y + h, x:x + w]

        # Predict mask status
        label = detect_mask(face)

        # Define color for bounding box
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
        