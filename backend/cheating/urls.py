"""Cheating app URLs.

This module registers API endpoints for the Cheating detection models.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    CheatingGroupsViewSet,
    CheatingGroupMembersViewSet,
    ConfirmedCheatersViewSet,
    FlaggedStudentsViewSet,
    SubmissionSimilarityPairsViewSet,
    LongitudinalCheatingGroupsViewSet,
    LongitudinalCheatingGroupMembersViewSet,
    LongitudinalCheatingGroupInstancesViewSet,
    similarity_plot,
    distribution_plot,
)

app_name = "cheating"

router = DefaultRouter()
router.register(r"cheating-groups", CheatingGroupsViewSet, basename="cheating-groups")
router.register(
    r"cheating-group-members",
    CheatingGroupMembersViewSet,
    basename="cheating-group-members",
)
router.register(
    r"confirmed-cheaters", ConfirmedCheatersViewSet, basename="confirmed-cheaters"
)
router.register(
    r"flagged-students", FlaggedStudentsViewSet, basename="flagged-students"
)
router.register(
    r"submission-similarity-pairs",
    SubmissionSimilarityPairsViewSet,
    basename="submission-similarity-pairs",
)
router.register(
    r"longitudinal-cheating-groups",
    LongitudinalCheatingGroupsViewSet,
    basename="longitudinal-cheating-groups",
)
router.register(
    r"longitudinal-cheating-group-members",
    LongitudinalCheatingGroupMembersViewSet,
    basename="longitudinal-cheating-group-members",
)
router.register(
    r"longitudinal-cheating-group-instances",
    LongitudinalCheatingGroupInstancesViewSet,
    basename="longitudinal-cheating-group-instances",
)

urlpatterns = [
    path("", include(router.urls)),
    path(
        "<int:assignment_id>/similarity-plot/",
        similarity_plot,
        name="similarity-plot"
    ),
    path(
        "<int:assignment_id>/distribution-plot/",
        distribution_plot,
        name="distribution-plot"
    ),
]
