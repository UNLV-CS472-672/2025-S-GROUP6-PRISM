"""Admin configuration for the assignments app."""

from django.contrib import admin
from assignments import models

# Register your models here.
admin.site.register(models.Student)
admin.site.register(models.FlaggedStudent)
admin.site.register(models.Submission)
admin.site.register(models.FlaggedSubmission)
admin.site.register(models.ConfirmedCheater)
admin.site.register(models.Assignment)
