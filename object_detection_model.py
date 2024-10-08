import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json
import os

# print("Current working directory:", os.getcwd())
# print("\nContents of the current directory:")
# for item in os.listdir():
#     print(item)

# print("\nContents of the 'coco_dataset' directory (if it exists):")
# coco_dir = os.path.join(os.getcwd(), 'coco_dataset')
# if os.path.exists(coco_dir):
#     for item in os.listdir(coco_dir):
#         print(item)
# else:
#     print("'coco_dataset' directory not found")


# Set the path to your COCO dataset
COCO_DIR = os.path.join(os.getcwd(), 'coco_dataset')  # Replace with your actual path

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load COCO dataset
with open(os.path.join(COCO_DIR, 'annotations', 'instances_val2017.json')) as f:
    data = json.load(f)

categories = {cat['id']: cat['name'] for cat in data['categories']}
NUM_CLASSES = len(categories)

# Create a base model from MobileNetV2
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')

# Add custom layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
output = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Custom data generator to load COCO images and labels
def coco_generator(data, batch_size, max_images=1000):
    image_count = 0
    while True:
        batch_images = []
        batch_labels = []
        while len(batch_images) < batch_size:
            if image_count >= max_images:
                image_count = 0
            img_data = data['images'][image_count]
            image_count += 1

            img_path = os.path.join(COCO_DIR, "val2017", img_data['file_name'])
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')  # Ensure 3 channels
                img = img.resize(IMG_SIZE)
                img = img_to_array(img)

                if img.shape != (IMG_SIZE[0], IMG_SIZE[1], 3):
                    continue  # Skip images with incorrect shape

                # Create label
                label = np.zeros(NUM_CLASSES)
                annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_data['id']]
                for ann in annotations:
                    category_id = ann['category_id']
                    if category_id in categories:
                        label[list(categories.keys()).index(category_id)] = 1

                batch_images.append(img)
                batch_labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

        yield np.array(batch_images), np.array(batch_labels)


# Create a tf.data.Dataset
def gen():
    return coco_generator(data, BATCH_SIZE)


dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
    )
)

# Train the model
model.fit(
    dataset,
    steps_per_epoch=1000 // BATCH_SIZE,  # Using 1000 images for this example
    epochs=EPOCHS
)

# Save the model
model.save('coco_model.h5')

# Convert the model to TensorFlow.js format
# Note: You need to have tensorflowjs installed (pip install tensorflowjs)
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, 'tfjs_model')
