"""Views for the Cheating app with enhanced filtering, ordering, search, and pagination."""

from rest_framework import filters, viewsets
from django_filters.rest_framework import DjangoFilterBackend
from prism_backend.mixins import CachedViewMixin

from .models import (
    CheatingGroups,
    CheatingGroupMembers,
    ConfirmedCheaters,
    FlaggedStudents,
    SubmissionSimiliarityPairs,
    LongitudinalCheatingGroups,
    LongitudinalCheatingGroupMembers,
    LongitudinalCheatingGroupInstances,
)
from .serializers import (
    CheatingGroupsSerializer,
    CheatingGroupMembersSerializer,
    ConfirmedCheatersSerializer,
    FlaggedStudentsSerializer,
    SubmissionSimiliarityPairsSerializer,
    LongitudinalCheatingGroupsSerializer,
    LongitudinalCheatingGroupMembersSerializer,
    LongitudinalCheatingGroupInstancesSerializer,
)
from .pagination import StandardResultsSetPagination


class CheatingGroupsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling CheatingGroups entries."""

    queryset = CheatingGroups.objects.all()
    serializer_class = CheatingGroupsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["assignment", "cohesion_score"]
    ordering_fields = ["cohesion_score"]
    ordering = ["cohesion_score"]
    search_fields = ["analysis_report_path", "assignment__title"]


class CheatingGroupMembersViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling CheatingGroupMembers entries."""

    queryset = CheatingGroupMembers.objects.all()
    serializer_class = CheatingGroupMembersSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["cheating_group", "student"]
    ordering_fields = ["cluster_distance"]
    ordering = ["cluster_distance"]
    search_fields = []


class ConfirmedCheatersViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling ConfirmedCheaters entries."""

    queryset = ConfirmedCheaters.objects.all()
    serializer_class = ConfirmedCheatersSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["confirmed_date", "assignment", "student"]
    ordering_fields = ["confirmed_date", "threshold_used"]
    ordering = ["confirmed_date"]
    search_fields = ["assignment__title", "student__first_name", "student__last_name"]


class FlaggedStudentsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling FlaggedStudents entries."""

    queryset = FlaggedStudents.objects.all()
    serializer_class = FlaggedStudentsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["professor", "student", "generative_ai"]
    ordering_fields = ["generative_ai"]
    ordering = ["generative_ai"]
    search_fields = [
        "professor__user__username",
        "student__first_name",
        "student__last_name",
    ]


class SubmissionSimiliarityPairsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling SubmissionSimiliarityPairs entries."""

    queryset = SubmissionSimiliarityPairs.objects.all()
    serializer_class = SubmissionSimiliarityPairsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["assignment", "file_name", "match_id"]
    ordering_fields = ["percentage"]
    ordering = ["percentage"]
    search_fields = ["file_name", "assignment__title"]


class LongitudinalCheatingGroupsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling LongitudinalCheatingGroups entries."""

    queryset = LongitudinalCheatingGroups.objects.all()
    serializer_class = LongitudinalCheatingGroupsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["score"]
    ordering_fields = ["score"]
    ordering = ["score"]
    search_fields = []


class LongitudinalCheatingGroupMembersViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling LongitudinalCheatingGroupMembers entries."""

    queryset = LongitudinalCheatingGroupMembers.objects.all()
    serializer_class = LongitudinalCheatingGroupMembersSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["longitudinal_cheating_group", "student", "is_core_member"]
    ordering_fields = ["appearance_count"]
    ordering = ["appearance_count"]
    search_fields = []


class LongitudinalCheatingGroupInstancesViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling LongitudinalCheatingGroupInstances entries."""

    queryset = LongitudinalCheatingGroupInstances.objects.all()
    serializer_class = LongitudinalCheatingGroupInstancesSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["cheating_group", "longitudinal_cheating_group"]
    ordering_fields = ["appearance_count"]
    ordering = ["appearance_count"]
    search_fields = []
