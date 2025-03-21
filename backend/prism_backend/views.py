from rest_framework.decorators import api_view
from rest_framework.response import Response
from google.oauth2 import id_token
from google.auth.transport import requests as google_auth_request
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


@api_view(["POST"])
def validate_token(self, request):
    token = request.data.get("token")

    if not token:
        return Response({"error": "Token is missing."}, status=400)

    try:
        decoded_token = id_token.verify_oauth2_token(
            id_token=token,
            request=google_auth_request.Request(),
            audience=GOOGLE_CLIENT_ID,
        )
        return Response(
            {"message": "Token is valid", "user": decoded_token}, status=200
        )
    except ValueError:
        return Response({"error": "Invalid token"}, status=401)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(["GET"])
def hello_world(request):
    return Response({"message": "Hello from Django Rest Framework!"})

