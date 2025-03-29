"""URLs mapping for the User API."""

from rest_framework.routers import DefaultRouter
from users.views import UserVS
from django.urls import path, include

# this automates the url so its cleaner and simpelr to work with, UserVS
# will take care of '/users/' URLs
user_router = DefaultRouter()
user_router.register("users", UserVS, basename="user")

urlpatterns = [
    path("", include(user_router.urls)),
]

# this will be used for the reverse mapping in 'test_user_api.py' file
# the line CREATE_USER_URL = reverse('user:user-list') will find the url because
# we define the app_name = user here (inside urls.py)
# so inside the 'user' namespace we are looking for 'users-list' if that
# makes snese
app_name = "user"
