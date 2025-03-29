"""Assignments URLs."""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

# from rest_framework_nested.routers import NestedDefaultRouter
from assignments.views import (
    AssignmentVS,
    SubmissionVS,
    FlaggedSubmissionVS,
    FlaggedStudentVS,
    ConfirmedCheaterVS,
    StudentVS,
)

router = DefaultRouter()
router.register("assignments", AssignmentVS, basename="assignments")
router.register("submissions", SubmissionVS, basename="submissions")
router.register(
    "flagged-submission", FlaggedSubmissionVS, basename="flagged-submission"
)
router.register("flagged-student", FlaggedStudentVS, basename="flagged-student")
router.register("confirmed-cheater", ConfirmedCheaterVS, basename="confirmed-cheater")
router.register("student", StudentVS, basename="student")

urlpatterns = [
    path("", include(router.urls)),
]
