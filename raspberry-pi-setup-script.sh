#!/bin/bash

# Update and upgrade the system
sudo apt update -y && sudo apt upgrade -y
if [ $? -ne 0 ]; then
    echo "Error during system update/upgrade. Please check the logs."
    exit 1
fi

# Install dependencies (excluding deprecated Qt4 packages)
sudo apt install -y python3-pip python3-dev libatlas-base-dev \
    libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 \
    libopenblas-dev libblas-dev liblapack-dev gfortran \
    libopencv-dev python3-opencv
if [ $? -ne 0 ]; then
    echo "Error during system package installation. Please check the logs."
    exit 1
fi

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose
pip3 install dlib flask flask-cors
if [ $? -ne 0 ]; then
    echo "Error during Python package installation. Please check the logs."
    exit 1
fi

# Check for Git installation (if cloning a repository)
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Installing Git..."
    sudo apt install -y git
    if [ $? -ne 0 ]; then
        echo "Error installing Git. Please check the logs."
        exit 1
    fi
fi

# Clone face_recognition repository from GitHub
if [ ! -d "face_recognition" ]; then
    echo "Cloning the face_recognition repository..."
    git clone https://github.com/ageitgey/face_recognition.git
    if [ $? -ne 0 ]; then
        echo "Error cloning face_recognition repository. Please check the logs."
        exit 1
    fi
else
    echo "face_recognition repository already exists."
fi

# Navigate to the face_recognition directory and install its dependencies
cd face_recognition || exit
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing dependencies for face_recognition. Please check the logs."
    exit 1
fi

# Install face_recognition_models from GitHub
pip3 install git+https://github.com/ageitgey/face_recognition_models
if [ $? -ne 0 ]; then
    echo "Error installing face_recognition_models. Please check the logs."
    exit 1
fi

# Go back to the project root directory
cd ..

# Clone your own GitHub repository (if it doesn't exist)
if [ ! -d "102_face_detection" ]; then
    echo "Cloning your repository..."
    git clone https://github.com/TrepidShark/102_face_detection.git
    if [ $? -ne 0 ]; then
        echo "Error cloning your repository. Please check the logs."
        exit 1
    fi
else
    echo "Your repository already exists."
fi

# Navigate to your repository
cd 102_face_detection || exit

# Final output messages
echo "Setup complete!"
echo "Don't forget to add your known faces to the 'known_faces' directory."
echo "To run the Flask server, execute: python3 facial_recognition_auth_v2.py"
echo "To start a simple web server for the HTML, execute: python3 -m http.server 8000"
