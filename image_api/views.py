import io
import joblib
import numpy as np
import tensorflow as tf

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


@csrf_exempt
def verify_image(request):

    if request.method != "POST":
        return JsonResponse({
            "success": False,
            "message": "Only POST requests are allowed."
        }, status=405)

    if "image" not in request.FILES:
        return JsonResponse({
            "success": False,
            "message": "No image was uploaded."
        }, status=400)

    image_file = request.FILES["image"]

    image = load_img(
        io.BytesIO(image_file.read()),
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
    print("Building:", result)

    return JsonResponse({
        "success": True,
        "result": result
    })