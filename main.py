import cv2
import face_recognition
import numpy as np
import os
import logging

# Set up logging to both console and file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create file handler
file_handler = logging.FileHandler('face_recognition.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



# Load images from the "faces" folder
KNOWN_FACES_DIR = "faces"
known_faces = []
known_names = []

# Load known images
for filename in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
    try:
        encoding = face_recognition.face_encodings(image)[0]  # Extract face encodings
    except IndexError:
        logging.warning(f"No faces found in the image: {filename}")
        continue
    except Exception as e:
        logging.error(f"Error processing image {filename}: {str(e)}")
        continue


    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])  # Use filename as the name

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if frame is None:
        logging.error("Failed to capture frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Detect faces in the frame
    try:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception as e:
        logging.error(f"Error during face detection: {str(e)}")
        continue


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Find best match
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
