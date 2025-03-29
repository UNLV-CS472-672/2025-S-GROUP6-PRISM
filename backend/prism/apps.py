"""This module defines the configuration for the 'prism' application in a Django project."""

from django.apps import AppConfig


class PrismConfig(AppConfig):
    """Configuration class for the 'prism' application.

    This class inherits from Django's AppConfig and is used to define
    application-specific settings for the 'prism' app.

    Attributes:
        default_auto_field (str): Specifies the type of primary key to use for models
            in this app. Defaults to "django.db.models.BigAutoField".
        name (str): The full Python path to the application, used for application
            identification. Set to "prism".
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "prism"
