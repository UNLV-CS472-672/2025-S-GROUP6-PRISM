"""Admin configuration for the courses app."""

from django.contrib import admin
from courses import models

# Register your models here.
admin.site.register(models.CourseCatalog)
admin.site.register(models.CourseInstances)
admin.site.register(models.Students)
admin.site.register(models.StudentEnrollments)
admin.site.register(models.Professors)
admin.site.register(models.ProfessorEnrollments)
admin.site.register(models.TeachingAssistants)
admin.site.register(models.TeachingAssistantEnrollment)
admin.site.register(models.Semester)
