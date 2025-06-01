from tensorflow.keras.models import load_model

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Đường dẫn dataset
dataset_path = "./archive1z"
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Data Augmentation
train_datagen = ImageDataGenerator(
    brightness_range=[0.8, 1.2],
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode="categorical"
)

model_path = "./model/fer2013_resnet_grayscale.h5"
model = load_model(model_path)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")