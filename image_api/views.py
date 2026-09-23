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


feature_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

feature_model.trainable = False

classifier = joblib.load(
    settings.BASE_DIR / "business_one_class.pkl"
)


def response(data, status=200):
    result = JsonResponse(data, status=status)
    result["Access-Control-Allow-Origin"] = "*"
    result["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    result["Access-Control-Allow-Headers"] = "Content-Type"
    return result


@csrf_exempt
def verify_image(request):

    if request.method == "OPTIONS":
        return response({})

    if request.method != "POST":
        return response(
            {"matching_status": False},
            status=405
        )

    try:
        if "image" in request.FILES:
            image_data = request.FILES["image"].read()

        else:
            data = json.loads(request.body)
            image_url = data.get("image_url")

            if not image_url:
                return response(
                    {"matching_status": False},
                    status=400
                )

            image_response = requests.get(
                image_url,
                timeout=15
            )

            image_response.raise_for_status()
            image_data = image_response.content

        image = load_img(
            io.BytesIO(image_data),
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

        return response({
            "matching_status": bool(prediction == 1)
        })

    except Exception as error:
        print("Error:", error)

        return response(
            {"matching_status": False},
            status=400
        )