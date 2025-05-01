"""Views module for authentication.

This module contains views for handling Google authentication
and custom login functionality.
"""

from .serializers import (
    GoogleAuthSerializer,
    LoginSerializer,
)
from .serializers import (
    GoogleAuthSerializer,
    LoginSerializer,
)
from dj_rest_auth.jwt_auth import set_jwt_cookies
from dj_rest_auth.views import LoginView as DJLoginView
from dj_rest_auth.views import LoginView as DJLoginView
from rest_framework.views import APIView
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status


class GoogleAuthView(APIView):
    """View for handling Google authentication.

    This view allows users to authenticate using Google OAuth and sets
    JWT cookies for access and refresh tokens upon successful authentication.

    Attributes:
        permission_classes (list): Permissions required to access this view.
        throttle_scope (str): Throttle scope for rate limiting.
        serializer_class (Serializer): Serializer for validating Google ID token.
    """

    permission_classes = [AllowAny]
    throttle_scope = "auth"
    serializer_class = GoogleAuthSerializer

    def post(self, request: Request):
        """Handle POST request to authenticate and set JWT cookies.

        Args:
            request (Request): The HTTP request object containing user credentials.

        Returns:
            Response: HTTP 200 OK response with JWT cookies on success.

        Raises:
            ValidationError: If the ID token is invalid or user not found.
        """
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        response = Response(data, status=status.HTTP_200_OK)
        set_jwt_cookies(
            response=response,
            access_token=data["access"],
            refresh_token=data["refresh"],
        )
        return response


class CustomLoginView(DJLoginView):
    """Custom login view with throttling.

    Extends the default LoginView to define a custom throttle scope.

    Attributes:
        throttle_scope (str): Scope used for throttling login attempts.
    """

    serializer_class = LoginSerializer

    def get_response(self):
        """
        Generate the HTTP response for a successful login.

        This method extends the default behavior of the `get_response` method
        from the parent `LoginView` to include additional user-specific data,
        such as the `professor_id`, in the response.

        Returns:
            Response: An HTTP response object containing the login data, including
            JWT tokens and user details with the `professor_id` field.
        """
        # Use the serializer's validated data
        serializer = self.get_serializer(data=self.request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # Get the base response from parent
        response = super().get_response()

        # Add extra data to the response
        response.data["user"]["professor_id"] = data.get("professor_id")

        return response
