"""Serializer for handling Google OAuth2 authentication."""

from google.oauth2 import id_token
from google.auth.transport import requests
from django.contrib.auth import get_user_model
from dj_rest_auth.serializers import LoginSerializer as DJLoginSerializer
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken
from datetime import datetime, timezone
from dotenv import load_dotenv
from courses.models import Professors
import os

load_dotenv()

User = get_user_model()


class GoogleAuthSerializer(serializers.Serializer):
    """Serializer for validating Google ID tokens and issuing JWT credentials.

    Attributes:
        id_token (serializers.CharField): A required field for the Google ID token.

    Methods:
        validate(attrs):
            Verifies the Google ID token and issues access and refresh JWT tokens
            if the user exists.
    """

    id_token = serializers.CharField()

    def validate(self, attrs: dict):
        """Validate the provided Google ID token and issue JWT tokens.

        This method verifies the authenticity of the provided Google ID token,
        checks if the user exists in the local database, and generates
        JWT access and refresh tokens for that user.

        Args:
            attrs (dict): A dictionary containing the field "id_token".

        Returns:
            dict: A dictionary containing access and refresh tokens,
            user details, and token expiration timestamps.

        Raises:
            serializers.ValidationError: If the token is invalid or the user
            does not exist.
        """
        id_token_str = attrs.get("id_token")

        try:
            google_info = id_token.verify_oauth2_token(
                id_token=id_token_str,
                request=requests.Request(),
                audience=os.getenv("GOOGLE_CLIENT_ID"),
            )

            if "email" not in google_info:
                raise serializers.ValidationError("Invalid Google token.")

            email = google_info["email"]
            user = User.objects.get(email=email)

            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token

            access_exp = (
                datetime.fromtimestamp(access_token["exp"], tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

            refresh_exp = (
                datetime.fromtimestamp(refresh["exp"], tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

            return {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user": {
                    "pk": user.pk,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                },
                "access_expiration": access_exp,
                "refresh_expiration": refresh_exp,
            }

        except ValueError:
            raise serializers.ValidationError("Invalid token.")


class LoginSerializer(DJLoginSerializer):
    """Extending the functionality of the LoginSerializer from dj-rest-auth to include the professor id in the response."""

    def validate(self, attrs):
        """Add professor_id to the JWT payload.

        Args:
            attrs (self, attrs): These self is simply for the request and attrs holds the
            login response object from LoginView from dj-rest-auth. It should include token information
            and user information.

        Returns:
            _type_: data object that contains the login response.
        """
        data = super().validate(attrs)

        user = data.get("user")

        try:
            professor = Professors.objects.get(user_id=user.id)
            data["professor_id"] = professor.id
        except Professors.DoesNotExist:
            print("FPSERES")
            data["professor_id"] = None

        return data


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Customize the process of obtaining JWT tokens by adding user-specific data.

    This serializer extends the functionality of the `TokenObtainPairSerializer` from
    the `rest_framework_simplejwt` package by including the `professor_id` in the JWT payload
    if the user is associated with a professor record.

    Methods:
        get_token(cls, user):
            Override the default `get_token` method to add the `professor_id` to the token payload.

    Attributes:
        None
    """

    @classmethod
    def get_token(cls, user):
        """
        Add the `professor_id` to the JWT payload if the user is associated with a professor record.

        Args:
            user (User): The user instance for whom the token is being generated.

        Returns:
            RefreshToken: A JWT token with additional `professor_id` field if applicable.
        """
        token = super().get_token(user)

        # Try to get professor_id and add it to the token
        try:
            professor = Professors.objects.get(user_id=user.id)
            token["professor_id"] = professor.id
        except Professors.DoesNotExist:
            token["professor_id"] = None

        return token
