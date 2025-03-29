"""asgi config for prism_backend."""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter
import prism_backend.routing  # Ensure it's imported

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prism_backend.settings")

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),  # Handles HTTP requests
        "websocket": prism_backend.routing.application,  # Ensure WebSockets are routed
    }
)
