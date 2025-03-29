"""App configuration for the Courses app."""

from django.apps import AppConfig


class CoursesConfig(AppConfig):
    """Django application configuration class for the 'courses' app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "courses"
