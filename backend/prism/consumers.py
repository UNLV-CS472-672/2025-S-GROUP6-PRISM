"""WebSocket consumer for real-time communication."""

from channels.generic.websocket import AsyncWebsocketConsumer
import json


class MyConsumer(AsyncWebsocketConsumer):
    """Asynchronous WebSocket consumer for handling WebSocket events.

    Methods:
        connect():
            Handles the connection event and sends a confirmation message.
        receive(text_data):
            Handles incoming messages and echoes the content.
        disconnect(close_code):
            Logs the disconnection event.

    Attributes:
        None
    """

    async def connect(self):
        """Handle WebSocket connection event.

        Accepts the connection and sends a JSON message to the client
        indicating successful connection.

        Sends:
            JSON message: {"message": "WebSocket connected"}
        """
        await self.accept()
        await self.send(text_data=json.dumps({"message": "WebSocket connected"}))

    async def receive(self, text_data):
        """Handle incoming WebSocket message.

        Echoes the received `text_data` back to the client in JSON format.

        Args:
            text_data (str): Text message received from the client.

        Returns:
            None
        """
        await self.send(text_data=json.dumps({"message": f"You sent: {text_data}"}))

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection event.

        Logs the disconnection using the provided close code.

        Args:
            close_code (int): Code indicating the reason for disconnection.

        Returns:
            None
        """
        print(f"WebSocket disconnected: {close_code}")
