"""ruting for websocket connections."""

from django.urls import re_path
from prism.consumers import MyConsumer  # Make sure this matches your consumer

websocket_urlpatterns = [
    re_path(r"ws/$", MyConsumer.as_asgi()),
]
