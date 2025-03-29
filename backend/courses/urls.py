"""URL Configuration for the courses app."""

from rest_framework.routers import DefaultRouter

# from rest_framework_nested.routers import NestedDefaultRouter
from courses.views import (
    ProfessorVS,
    ProfessorClassSectionVS,
    SemesterVS,
    ClassVS,
    EnrollmentVS,
)
from django.urls import path, include

router = DefaultRouter()
router.register("classes", ClassVS, basename="class")
router.register("professors", ProfessorVS, basename="professor")
router.register("semesters", SemesterVS, basename="semester")
router.register(
    "sectionclassprof", ProfessorClassSectionVS, basename="sectionclassprof"
)
router.register("enrollments", EnrollmentVS, basename="enrollment")

urlpatterns = [
    path("", include(router.urls)),
]
