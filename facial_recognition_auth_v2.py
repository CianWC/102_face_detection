from flask import Flask, request, jsonify
from flask_cors import CORS
from face_recognition import load_image_file, face_locations, face_encodings, compare_faces
import numpy as np
import base64
import os
import cv2
import logging

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load known faces
known_face_encodings = []
known_face_names = []


def load_known_faces(directory):
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    logger.info(f"Attempting to load known faces from directory: {directory}")

    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return

    for person_name in os.listdir(directory):
        person_folder = os.path.join(directory, person_name)
        logger.debug(f"Checking folder: {person_folder}")

        if os.path.isdir(person_folder):
            logger.debug(f"Processing directory: {person_folder}")
            for filename in os.listdir(person_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_folder, filename)
                    logger.debug(f"Processing image: {image_path}")
                    try:
                        image = load_image_file(image_path)
                        logger.debug(f"Image loaded successfully: {image_path}")

                        face_encodings_result = face_encodings(image)
                        if face_encodings_result:
                            encoding = face_encodings_result[0]
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                            logger.info(f"Successfully encoded face for {person_name} from {filename}")
                        else:
                            logger.warning(f"No face found in {filename} for {person_name}")
                    except Exception as e:
                        logger.error(f"Error processing {filename} for {person_name}: {str(e)}")
        else:
            logger.warning(f"Not a directory: {person_folder}")

    logger.info(f"Loaded {len(known_face_names)} known faces")


# Load known faces from the directory
load_known_faces("known_faces")

@app.route('/')  # Add this route
def index():
    return "Welcome to the Facial Recognition App!" 

@app.route('/detect', methods=['POST'])
def detect():
    logger.debug("Received detection request")

    # Get the image data from the request
    image_data = request.json['image']

    # Decode the base64 image
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Use OpenCV to decode the image array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    logger.debug(f"Received image shape: {image.shape}")

    # Convert image to RGB (OpenCV loads in BGR format)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations_frame = face_locations(rgb_frame)
    face_encodings_frame = face_encodings(rgb_frame, face_locations_frame)

    logger.debug(f"Detected {len(face_locations_frame)} faces")

    # Initialize the results
    results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations_frame, face_encodings_frame):
        # See if the face is a match for the known face(s)
        matches = compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        results.append({
            "name": name,
            "bbox": [left, top, right, bottom]
        })

    logger.debug(f"Results: {results}")
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
