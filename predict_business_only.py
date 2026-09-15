import os
import numpy as np
import joblib

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

feature_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

feature_model.trainable = False


classifier = joblib.load("business_one_class.pkl")


image_path = input("Enter image path: ").strip().strip('"')


if not os.path.exists(image_path):
    print("Image not found.")
    exit()


image = load_img(
    image_path,
    target_size=(224, 224)
)

image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

image_array = preprocess_input(image_array)


features = feature_model.predict(
    image_array,
    verbose=0
)
prediction = classifier.predict(features)[0]


if prediction == 1:
    result = "YES"
else:
    result = "NO"


print("Prediction:", prediction)
print("Business Building:", result)