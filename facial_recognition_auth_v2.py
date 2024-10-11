from flask import Flask, request, jsonify
from flask_cors import CORS
from face_recognition import load_image_file, face_locations, face_encodings, compare_faces
import numpy as np
import base64
import os
import cv2

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Load known faces
known_face_encodings = []
known_face_names = []
frame_count = 0  # Global counter to track frames


def load_known_faces(directory):
    for person_name in os.listdir(directory):
        person_folder = os.path.join(directory, person_name)
        if os.path.isdir(person_folder):  # Check if it's a directory
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image = load_image_file(os.path.join(person_folder, filename))
                    encoding = face_encodings(image)
                    if encoding:  # Check if encoding is found
                        known_face_encodings.append(encoding[0])
                        known_face_names.append(person_name)  # Use the folder name as the person's name


# Load known faces from the directory
load_known_faces("known_faces")


@app.route('/detect', methods=['POST'])
def detect():
    global frame_count  # Access the global frame counter

    # Increment the frame counter
    frame_count += 1

    # Skip every n frames (e.g., only process every 5th frame)
    if frame_count % 30 != 0:
        return jsonify({"status": "skipped"})

    # Get the image data from the request
    image_data = request.json['image']

    # Decode the base64 image
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Use OpenCV to decode the image array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert image to RGB (OpenCV loads in BGR format)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations_frame = face_locations(rgb_frame)
    face_encodings_frame = face_encodings(rgb_frame, face_locations_frame)

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

    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
