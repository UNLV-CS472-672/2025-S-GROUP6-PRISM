"""App configuration for the data_ingestion app."""

from django.apps import AppConfig


class DataIngestionConfig(AppConfig):
    """Configuration class for the 'data_ingestion' application."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "data_ingestion"
