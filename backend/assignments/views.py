"""Assignment views with enhanced filtering, ordering, and search capabilities."""

from rest_framework import filters, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from rest_framework import status
from django_filters.rest_framework import DjangoFilterBackend
from prism_backend.mixins import CachedViewMixin
from users.permissions import IsProfessorOrAdmin
from django.db.models import Count, Max, Avg, F, Q
from cheating.models import SubmissionSimilarityPairs
from courses.models import CourseInstances

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
    AssignmentCreateSerializer,
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

    def get_serializer_class(self):
        """Return the appropriate serializer class based on the action.

        Uses AssignmentCreateSerializer for 'create' actions to handle
        nested writes. Defaults to AssignmentsSerializer for all other actions.

        Returns:
            serializers.ModelSerializer: The serializer class to use.
        """
        if self.action == "create":
            return AssignmentCreateSerializer
        return AssignmentsSerializer

    @action(detail=False, methods=["get"], url_path="get-assignments-by-course")
    def get_assignments_by_course(self, request: Request) -> Response:
        """Return all assignments for a given course instance ID.

        Example: /assignments/get-assignments-by-course/?course=12
        """
        courseinstance_id = request.query_params.get("course")

        if not courseinstance_id:
            return Response(
                {"detail": "'course' query parameter is required."},
                status=400,
            )

        try:
            course_instance = CourseInstances.objects.select_related(
                "course_catalog", "semester"
            ).get(id=courseinstance_id)
        except CourseInstances.DoesNotExist:
            return Response({"detail": "CourseInstance not found."}, status=404)

        # Fetch assignments matching course catalog and semester
        assignments = Assignments.objects.filter(
            course_catalog=course_instance.course_catalog,
            semester=course_instance.semester,
        ).order_by("assignment_number")

        page = self.paginate_queryset(assignments)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(assignments, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


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

    def get_queryset(self):
        """Return submissions filtered by logical AND on query parameters.

        Supports filtering by one or both semesters.

        Example query:
            /submissions/?student=382&asid=1&course=3&semester1=1&semester2=2

        Returns:
            QuerySet: A filtered Django QuerySet of Submissions.
        """
        queryset = Submissions.objects.select_related(
            "assignment", "course_instance", "student"
        ).all()

        assignment_id = self.request.query_params.get("asid")
        course_instance_id = self.request.query_params.get("course")
        semester1 = self.request.query_params.get("semester")
        semester2 = self.request.query_params.get("semester2")
        student_id = self.request.query_params.get("student")

        if assignment_id:
            queryset = queryset.filter(assignment_id=assignment_id)

        if course_instance_id:
            queryset = queryset.filter(course_instance_id=course_instance_id)

        if semester1 and semester2:
            queryset = queryset.filter(
                Q(assignment__semester_id=semester1)
                | Q(assignment__semester_id=semester2)
            )
        elif semester1:
            queryset = queryset.filter(assignment__semester_id=semester1)
        elif semester2:
            queryset = queryset.filter(assignment__semester_id=semester2)

        if student_id:
            queryset = queryset.filter(student_id=student_id)

        return queryset


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


class AggregatedAssignmentDataView(APIView):
    """
    API endpoint for aggregated assignment data.

    Returns combined statistics (i.e. similarity scores and flagged counts)
    for submissions, grouped and filtered as requested. Professors, TAs, and
    admins can filter by students, assignments, and course instance.
    """

    permission_classes = [IsAuthenticated, IsProfessorOrAdmin]

    def get(self, request, format=None):
        """Return aggregated similarity statistics for submissions."""
        student_ids = request.query_params.getlist("students")
        assignment_ids = request.query_params.getlist("assignments")
        course_id = request.query_params.get("course")

        # Build a single base queryset and apply any filters
        base_qs = SubmissionSimilarityPairs.objects.select_related(
            "assignment",
            "submission_id_1__student",
            "submission_id_2__student",
        )

        if student_ids:
            base_qs = base_qs.filter(submission_id_1__student__id__in=student_ids)
        if assignment_ids:
            base_qs = base_qs.filter(assignment__id__in=assignment_ids)
        if course_id:
            base_qs = base_qs.filter(submission_id_1__course_instance_id=course_id)

        response_data = {}

        # Highest similarity score per student
        response_data["student_max_similarity_score"] = list(
            base_qs.values(
                "submission_id_1__student__id",
                "submission_id_1__student__first_name",
                "submission_id_1__student__last_name",
            ).annotate(max_score=Max("percentage"))
        )

        # Average similarity score per assignment
        response_data["assignment_avg_similarity_score"] = list(
            base_qs.values(
                "assignment__id",
                "assignment__title",
            ).annotate(avg_score=Avg("percentage"))
        )

        # Flagged submissions per assignment (uses dynamic threshold)
        response_data["flagged_per_assignment"] = list(
            base_qs.filter(
                percentage__gte=F("assignment__required_files__similarity_threshold")
            )
            .values("assignment__title")
            .annotate(flagged_count=Count("id"))
        )

        # Similarity score trend over time (per submission date)
        response_data["similarity_trends"] = list(
            base_qs.values(date=F("submission_id_1__created_at"))
            .annotate(avg_similarity=Avg("percentage"))
            .order_by("date")
        )

        # Flagged submissions per professor
        response_data["flagged_by_professor"] = list(
            base_qs.filter(
                percentage__gte=F("assignment__required_files__similarity_threshold")
            )
            .values("submission_id_1__course_instance__professor__user__username")
            .annotate(flagged_count=Count("id"))
        )

        # Professor wise average similarity score
        response_data["professor_avg_similarity"] = list(
            base_qs.values(
                "submission_id_1__course_instance__professor__user__username"
            ).annotate(avg_similarity=Avg("percentage"))
        )

        return Response(response_data, status=200)
