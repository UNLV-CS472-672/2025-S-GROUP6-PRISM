"""admin.py file for the users app."""

from django.contrib import admin
from users import models

# Register your models here.
admin.site.register(models.User)
