"""Courses app URLs.

This module registers API endpoints for the core Courses app models.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    CourseCatalogViewSet,
    CourseInstancesViewSet,
    SemesterViewSet,
    StudentsViewSet,
    StudentEnrollmentsViewSet,
    ProfessorsViewSet,
    ProfessorEnrollmentsViewSet,
    TeachingAssistantsViewSet,
    TeachingAssistantEnrollmentViewSet,
)

router = DefaultRouter()
router.register(r"coursecatalog", CourseCatalogViewSet, basename="coursecatalog")
router.register(r"courseinstances", CourseInstancesViewSet, basename="courseinstances")
router.register(r"semester", SemesterViewSet, basename="semester")
router.register(r"students", StudentsViewSet, basename="students")
router.register(
    r"studentenrollments", StudentEnrollmentsViewSet, basename="studentenrollments"
)
router.register(r"professors", ProfessorsViewSet, basename="professors")
router.register(
    r"professorenrollments",
    ProfessorEnrollmentsViewSet,
    basename="professorenrollments",
)
router.register(
    r"teachingassistants", TeachingAssistantsViewSet, basename="teachingassistants"
)
router.register(
    r"teachingassistantenrollment",
    TeachingAssistantEnrollmentViewSet,
    basename="teachingassistantenrollment",
)

urlpatterns = [
    path("", include(router.urls)),
]
