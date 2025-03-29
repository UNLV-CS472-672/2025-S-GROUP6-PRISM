"""Admin configuration for the courses app."""

from django.contrib import admin
from courses import models

# Register your models here.
admin.site.register(models.Professor)
admin.site.register(models.Class)
admin.site.register(models.Semester)
admin.site.register(models.ProfessorClassSection)
admin.site.register(models.Enrollment)
