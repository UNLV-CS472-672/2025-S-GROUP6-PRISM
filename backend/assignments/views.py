"""Assignment views with enhanced filtering, ordering, and search capabilities."""

from rest_framework import filters, viewsets
from django_filters.rest_framework import DjangoFilterBackend
from prism_backend.mixins import CachedViewMixin

from .models import (
    Assignments,
    Submissions,
    BaseFiles,
    BulkSubmissions,
    Constraints,
    PolicyViolations,
    RequiredSubmissionFiles,
)
from .pagination import StandardResultsSetPagination
from .serializers import (
    AssignmentsSerializer,
    SubmissionsSerializer,
    BaseFilesSerializer,
    BulkSubmissionsSerializer,
    ConstraintsSerializer,
    PolicyViolationsSerializer,
    RequiredSubmissionFilesSerializer,
)


class AssignmentsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling Assignments."""

    queryset = Assignments.objects.all()
    serializer_class = AssignmentsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = [
        "course_catalog",
        "semester",
        "assignment_number",
        "language",
        "has_base_code",
        "has_policy",
        "due_date",
    ]
    ordering_fields = ["assignment_number", "due_date"]
    ordering = ["assignment_number"]
    search_fields = [
        "title",
        "pdf_filepath",
        "moss_report_directory_path",
    ]


class SubmissionsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling Submissions."""

    queryset = Submissions.objects.all()
    serializer_class = SubmissionsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["grade", "flagged", "course_instance"]
    ordering_fields = ["grade", "created_at"]
    ordering = ["created_at"]
    search_fields = ["file_path"]


class BaseFilesViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling BaseFiles."""

    queryset = BaseFiles.objects.all()
    serializer_class = BaseFilesSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["file_name"]
    ordering_fields = ["file_name"]
    ordering = ["file_name"]
    search_fields = ["file_name", "file_path"]


class BulkSubmissionsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling BulkSubmissions."""

    queryset = BulkSubmissions.objects.all()
    serializer_class = BulkSubmissionsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["directory_path", "course_instance"]
    ordering_fields = ["directory_path"]
    ordering = ["directory_path"]
    search_fields = ["directory_path"]


class ConstraintsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling Constraints."""

    queryset = Constraints.objects.all()
    serializer_class = ConstraintsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = [
        "identifier",
        "is_library",
        "is_keyword",
        "is_permitted",
    ]
    ordering_fields = ["identifier"]
    ordering = ["identifier"]
    search_fields = ["identifier"]


class PolicyViolationsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling PolicyViolations."""

    queryset = PolicyViolations.objects.all()
    serializer_class = PolicyViolationsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["line_number"]
    ordering_fields = ["line_number"]
    ordering = ["line_number"]
    search_fields = []


class RequiredSubmissionFilesViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling RequiredSubmissionFiles."""

    queryset = RequiredSubmissionFiles.objects.all()
    serializer_class = RequiredSubmissionFilesSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["file_name"]
    ordering_fields = ["file_name"]
    ordering = ["file_name"]
    search_fields = ["file_name"]
