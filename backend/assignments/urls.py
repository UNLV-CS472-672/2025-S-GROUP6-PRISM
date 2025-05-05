"""Assignments URLs for the new viewsets.

This module registers API endpoints for:
- Assignments
- Submissions
- BaseFiles
- BulkSubmissions
- Constraints
- PolicyViolations
- RequiredSubmissionFiles
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    AssignmentsViewSet,
    SubmissionsViewSet,
    BaseFilesViewSet,
    BulkSubmissionsViewSet,
    ConstraintsViewSet,
    PolicyViolationsViewSet,
    RequiredSubmissionFilesViewSet,
    AggregatedAssignmentDataView,
)

router = DefaultRouter()
router.register(r"assignments", AssignmentsViewSet, basename="assignments")
router.register(r"submissions", SubmissionsViewSet, basename="submissions")
router.register(r"basefiles", BaseFilesViewSet, basename="basefiles")
router.register(r"bulksubmissions", BulkSubmissionsViewSet, basename="bulksubmissions")
router.register(r"constraints", ConstraintsViewSet, basename="constraints")
router.register(
    r"policyviolations", PolicyViolationsViewSet, basename="policyviolations"
)
router.register(
    r"requiredsubmissionfiles",
    RequiredSubmissionFilesViewSet,
    basename="requiredsubmissionfiles",
)

urlpatterns = [
    path("", include(router.urls)),
    path(
        "aggregated-data/",
        AggregatedAssignmentDataView.as_view(),
        name="aggregated-assignment-data",
    ),
]
