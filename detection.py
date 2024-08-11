import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Load the pre-trained model
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y']
model = load_model("sign_language.keras")


def classify(image):
    # Resize and normalize the image
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the gesture
    proba = model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    if not ret:
        break

    # Flip the image horizontally for a mirror effect
    img = cv2.flip(img, 1)

    # Define the region of interest (ROI) for gesture detection
    top, right, bottom, left = 75, 350, 300, 590
    roi = img[top:bottom, right:left]

    # Flip the ROI and convert it to grayscale
    roi = cv2.flip(roi, 1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Display the ROI
    cv2.imshow('ROI', gray)

    # Classify the gesture
    alpha = classify(gray)

    # Draw the ROI and display the prediction on the main video frame
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, alpha, (10, 130), font, 5, (0, 0, 255), 2)

    # Display the video frame with the gesture prediction
    cv2.imshow('Video', img)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()