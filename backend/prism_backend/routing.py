"""routing for websocket connections."""

from channels.routing import ProtocolTypeRouter, URLRouter
from prism.routing import websocket_urlpatterns

application = ProtocolTypeRouter(
    {
        "websocket": URLRouter(websocket_urlpatterns),
    }
)
