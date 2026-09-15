import os
import numpy as np
import joblib

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array


# Load MobileNetV2 feature extractor
feature_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

feature_model.trainable = False


# Load Business-only One-Class SVM
classifier = joblib.load("business_one_class.pkl")


# Ask for image
image_path = input("Enter image path: ").strip().strip('"')


# Check file
if not os.path.exists(image_path):
    print("Image not found.")
    exit()


# Load image
image = load_img(
    image_path,
    target_size=(224, 224)
)

image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

# Same preprocessing used during training
image_array = preprocess_input(image_array)


# Extract MobileNetV2 features
features = feature_model.predict(
    image_array,
    verbose=0
)


# Business-only prediction
prediction = classifier.predict(features)[0]


if prediction == 1:
    result = "YES"
else:
    result = "NO"


print("Prediction:", prediction)
print("Business Building:", result)