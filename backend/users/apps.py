"""users.apps - Configuration for the 'users' application in a Django project."""

from django.apps import AppConfig


class UsersConfig(AppConfig):
    """Configuration class for the 'users' application in a Django project.

    This class defines the default primary key field type for models in the app
    and specifies the name of the application.

    Attributes:
        default_auto_field (str): Specifies the type of primary key to use for models
            by default. In this case, it is set to "django.db.models.BigAutoField".
        name (str): The name of the application. Here, it is set to "users".
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "users"
