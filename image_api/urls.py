from django.urls import path
from .views import verify_image

urlpatterns = [
    path("verify/", verify_image, name="verify_image"),
]