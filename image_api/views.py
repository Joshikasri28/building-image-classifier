import io
import json

import joblib
import numpy as np
import requests

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array


# MobileNetV2 for feature extraction
feature_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

feature_model.trainable = False


# Load trained One-Class SVM model
classifier = joblib.load(
    settings.BASE_DIR / "business_one_class.pkl"
)


@csrf_exempt
def verify_image(request):

    if request.method != "POST":
        return JsonResponse(
            {"matching_status": False},
            status=405
        )

    try:
        # Read JSON request
        data = json.loads(request.body)

        image_url = data.get("image_url")

        if not image_url:
            return JsonResponse(
                {"matching_status": False},
                status=400
            )

        # Download image from URL
        response = requests.get(
            image_url,
            timeout=15
        )

        response.raise_for_status()

        # Load and resize image
        image = load_img(
            io.BytesIO(response.content),
            target_size=(224, 224)
        )

        # Convert image to array
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        # MobileNetV2 preprocessing
        image_array = preprocess_input(image_array)

        # Extract image features
        features = feature_model.predict(
            image_array,
            verbose=0
        )

        # One-Class SVM prediction
        prediction = classifier.predict(features)[0]

        matching_status = prediction == 1

        return JsonResponse({
            "matching_status": bool(matching_status)
        })

    except Exception as e:

        print("Error:", str(e))

        return JsonResponse({
            "matching_status": False
        }, status=400)