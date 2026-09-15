import os
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.svm import OneClassSVM


# Dataset path
DATASET_PATH = r"C:\Users\ELCOT\Desktop\Business\dataset\building"


# MobileNetV2 feature extractor
feature_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

feature_model.trainable = False


# Extract features
features = []
image_count = 0

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):

            image_path = os.path.join(root, file)

            try:
                image = load_img(
                    image_path,
                    target_size=(224, 224)
                )

                image_array = img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)
                image_array = preprocess_input(image_array)

                feature = feature_model.predict(
                    image_array,
                    verbose=0
                )

                features.append(feature[0])
                image_count += 1

            except Exception as e:
                print("Skipped:", image_path)
                print("Reason:", e)


features = np.array(features)

print("Business images:", image_count)
print("Feature shape:", features.shape)


# One-Class SVM
classifier = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.01
)

classifier.fit(features)


# Save model
joblib.dump(
    classifier,
    "business_one_class.pkl"
)

# Save features for reference
np.save(
    "business_features.npy",
    features
)

print("Business-only model training completed.")
print("Model saved as: business_one_class.pkl")