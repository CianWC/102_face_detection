#!/bin/bash

# Update and upgrade the system
sudo apt update -y
sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-dev libatlas-base-dev
sudo apt install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libopenblas-dev libblas-dev liblapack-dev
sudo apt install -y gfortran
sudo apt install -y libopencv-dev python3-opencv

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose
pip3 install dlib
pip3 install flask flask-cors

# Check for pip install errors
if [ $? -ne 0 ]; then
    echo "Error during Python package installation. Please check the logs."
    exit 1
fi

# Check for Git installation (if cloning a repository)
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Installing Git..."
    sudo apt install -y git
fi

# Clone face_recognition repository from GitHub
if [ ! -d "face_recognition" ]; then
    echo "Cloning the face_recognition repository..."
    git clone https://github.com/ageitgey/face_recognition.git
else
    echo "face_recognition repository already exists."
fi

# Navigate to the face_recognition directory and install its dependencies
cd face_recognition
pip3 install -r requirements.txt

# Check for any errors in installing face_recognition dependencies
if [ $? -ne 0 ]; then
    echo "Error installing dependencies for face_recognition. Please check the logs."
    exit 1
fi

# Install face_recognition_models from GitHub
pip3 install git+https://github.com/ageitgey/face_recognition_models

# Check if face_recognition_models installation was successful
if [ $? -ne 0 ]; then
    echo "Error installing face_recognition_models. Please check the logs."
    exit 1
fi

# Go back to the project root directory
cd ..

# Optionally clone your own GitHub repository (uncomment and modify the next line with your repo URL)
git clone https://github.com/TrepidShark/102_face_detection.git
cd 102_face_detection


# Final output messages
echo "Setup complete!"
echo "Don't forget to add your known faces to the 'known_faces' directory."
echo "To run the Flask server, execute: python3 facial_recognition_auth_v2.py"
echo "To start a simple web server for the HTML, execute: python3 -m http.server 8000"
