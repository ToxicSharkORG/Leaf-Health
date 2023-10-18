from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import keyboard

# Function to capture an image
def capture_image():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Could not read frame.")
        return

    # Save the captured frame as an image file
    cv2.imwrite("captured_image.jpg", frame)

    # Release the camera
    cap.release()

    print("Image captured and saved as 'captured_image.jpg'")

def classify_leaf(image_path, leaf_model, disease_model, class_names_leaf, class_names_disease):
    # Load and process the captured image
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict if it's a leaf using the leaf model
    prediction_leaf = leaf_model.predict(data)
    index_leaf = np.argmax(prediction_leaf)
    class_name_leaf = class_names_leaf[index_leaf]

    # Print leaf prediction and confidence score
    print("Leaf Class:", class_name_leaf, end=" ")
    print("Leaf Confidence Score:", prediction_leaf[0][index_leaf])

    if class_name_leaf == 'LEAF':
        # If it's classified as a leaf, use the disease model
        prediction_disease = disease_model.predict(data)
        index_disease = np.argmax(prediction_disease)
        class_name_disease = class_names_disease[index_disease]

        # Print disease prediction and confidence score
        print("Disease Class:", class_name_disease, end=" ")
        print("Disease Confidence Score:", prediction_disease[0][index_disease])

if __name__ == "__main__":
    while True:
        try:
            # Listen for a specific key press event (e.g., 'c' key)
            keyboard.on_press_key("c", lambda e: capture_image())
            print("Press 'c' to capture an image...")

            # Wait for the 'esc' key to exit the program
            keyboard.wait("esc")
        except KeyboardInterrupt:
            pass
        finally:
            keyboard.unhook_all()

        # Load the leaf model
        leaf_model = load_model("keras_Modelleafnoleaf.h5", compile=False)
        class_names_leaf = open("labelsleafno.txt", "r").readlines()

        # Load the disease model
        disease_model = load_model("keras_model.h5", compile=False)
        class_names_disease = open("labelsdisease.txt", "r").readlines()

        # Classify the captured image
        classify_leaf("captured_image.jpg", leaf_model, disease_model, class_names_leaf, class_names_disease)
