"""Serializer for handling Google OAuth2 authentication."""

from google.oauth2 import id_token
from google.auth.transport import requests
from django.contrib.auth import get_user_model
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken
from datetime import datetime, timezone
from dotenv import load_dotenv
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

            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                raise serializers.ValidationError(
                    "No account exists for this Google user."
                )

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
