import os
import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.svm import OneClassSVM


train_path = "data/train/building"

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

model.trainable = False


def extract_features(folder):

    features = []

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder, file)

        image = load_img(
            image_path,
            target_size=(224, 224)
        )

        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        feature = model.predict(image_array, verbose=0)[0]
        features.append(feature)

    return np.array(features)


print("Extracting building features...")

features = extract_features(train_path)

print("Building images:", len(features))

classifier = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

classifier.fit(features)

joblib.dump(classifier, "building_one_class.pkl")

np.save("building_features.npy", features)

print("Building-only model training completed.")