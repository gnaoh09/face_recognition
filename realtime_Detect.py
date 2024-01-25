import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
import os
from face_detect import labels as lb
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the saved Keras model
loaded_model = load_model("face.keras")
print("Model loaded.")

video_capture = cv2.VideoCapture(0)

class_labels = ['hoang', 'A', 'B']

while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    # Preprocess the frame (resize to match the model input size and normalize)
    input_frame = cv2.resize(video_frame, (32, 32))
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
    input_frame = input_frame / 255.0  # Normalize

    # Use the loaded model to make predictions
    predictions = loaded_model.predict(input_frame)
    predicted_class = np.argmax(predictions)

    print("Predicted Class Index:", predicted_class)
    print("Number of Class Labels:", len(class_labels))


    predicted_label = class_labels[predicted_class-1]


    print("Predicted Label:", predicted_label)

    # Display the predicted class as text on the frame
    cv2.putText(
        video_frame,
        f"Predicted Class: {predicted_class}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        video_frame,
        f"Predicted Label: {predicted_label}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Display the frame with predictions
    cv2.imshow("Face Recognition", video_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()