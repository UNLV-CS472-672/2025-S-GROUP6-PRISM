"""Assignment views with enhanced filtering, ordering, and search capabilities."""

from assignments import serializers
from assignments import models
from rest_framework import viewsets
from django_filters import rest_framework as filters
from rest_framework.filters import OrderingFilter, SearchFilter

from rest_framework.permissions import IsAuthenticated
from assignments.models import FlaggedSubmission
from assignments.serializers import FlaggedSubmissionSerializer
from users.permissions import is_admin


class StudentVS(viewsets.ModelViewSet):
    """
    Provide CRUD operations for the Student model.

    Supports filtering, ordering, and searching functionality.
    """

    queryset = models.Student.objects.all()
    serializer_class = serializers.StudentSerializer
    filter_backends = [filters.DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = {"id": ["exact"], "first_name": ["exact", "icontains"]}
    ordering_fields = ["id", "first_name"]
    ordering = ["first_name"]
    search_fields = ["first_name"]


class FlaggedStudentVS(viewsets.ModelViewSet):
    """
    Provide CRUD operations for FlaggedStudent objects.

    Supports filtering, ordering, and search.
    """

    queryset = models.FlaggedStudent.objects.all()
    serializer_class = serializers.FlaggedStudentSerializer
    filter_backends = [filters.DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = {
        "professor__id": ["exact"],
        "times_over_threshold": ["exact", "gte", "lte"],
    }
    ordering_fields = ["professor__user__first_name", "times_over_threshold"]
    ordering = ["times_over_threshold"]


class AssignmentVS(viewsets.ModelViewSet):
    """
    Provide CRUD operations for Assignment objects.

    Supports filtering, ordering, and search capabilities.
    """

    queryset = models.Assignment.objects.all()
    serializer_class = serializers.AssignmentSerializer
    filter_backends = [filters.DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = {
        "professor__id": ["exact"],
        "class_instance__name": ["exact", "icontains"],
        "title": ["exact", "icontains"],
        "assignment_number": ["exact"],
    }
    ordering_fields = [
        "professor__user__first_name",
        "class_instance__name",
        "title",
        "assignment_number",
    ]
    ordering = ["assignment_number"]
    search_fields = [
        "title",
        "class_instance__name",
        "professor__user__first_name",
    ]


class SubmissionVS(viewsets.ModelViewSet):
    """
    Provide CRUD operations for Submission objects.

    Supports filtering, ordering, and search functionality.
    """

    queryset = models.Submission.objects.all()
    serializer_class = serializers.SubmissionSerializer
    filter_backends = [filters.DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = {
        "student__id": ["exact"],
        "professor__id": ["exact"],
        "flagged": ["exact"],
        "assignment__class_instance__name": ["exact", "icontains"],
        "assignment__title": ["exact", "icontains"],
    }
    ordering_fields = [
        "student__first_name",
        "professor__user__first_name",
        "flagged",
        "assignment__class_instance__name",
        "assignment__title",
    ]
    ordering = ["assignment__title"]
    search_fields = ["assignment__title", "assignment__class_instance__name"]


class FlaggedSubmissionVS(viewsets.ModelViewSet):
    """
    Provide CRUD operations for FlaggedSubmission objects.

    Supports filtering, ordering, and searching functionality.
    """

    queryset = models.FlaggedSubmission.objects.all()
    serializer_class = serializers.FlaggedSubmissionSerializer
    filter_backends = [filters.DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = {
        "submission__professor__id": ["exact"],
        "file_name": ["exact", "icontains"],
        "percentage": ["exact", "gte", "lte"],
    }
    ordering_fields = [
        "submission__professor__user__first_name",
        "file_name",
        "percentage",
    ]
    ordering = ["-percentage"]
    search_fields = ["file_name", "submission__professor__user__first_name"]


class ConfirmedCheaterVS(viewsets.ModelViewSet):
    """
    Provide CRUD operations for ConfirmedCheater objects.

    Supports filtering, ordering, and searching.
    """

    queryset = models.ConfirmedCheater.objects.all()
    serializer_class = serializers.ConfirmedCheaterSerializer
    filter_backends = [filters.DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = {
        "professor__id": ["exact"],
        "confirmed_date": ["exact", "gte", "lte"],
        "threshold_used": ["exact", "gte", "lte"],
    }
    ordering_fields = [
        "professor__user__first_name",
        "confirmed_date",
        "threshold_used",
    ]
    ordering = ["confirmed_date"]


class PlagiarismReportViewSet(viewsets.ModelViewSet):
    """
    Provide flagged submission reports based on user access level.

    Admins see all, professors see their own.
    """

    serializer_class = FlaggedSubmissionSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.DjangoFilterBackend, OrderingFilter, SearchFilter]
    filterset_fields = [
        "submission__assignment__class_instance_id",
        "submission__assignment__class_instance__semester",
        "submission__assignment__class_instance__profclassect__semester_id",
    ]
    ordering_fields = [
        "submission__created_at",
        "submission__professor__user__first_name",
        "file_name",
        "percentage",
    ]
    search_fields = ["file_name", "submission__professor__user__first_name"]

    def get_queryset(self):
        """
        Return the queryset of flagged submissions based on user's role.

        Admins see all, professors see only their own.
        """
        user = self.request.user

        if is_admin(user):
            return FlaggedSubmission.objects.all()
        else:
            professor = user.user_professor
            return FlaggedSubmission.objects.filter(
                submission__professor=professor
            ).distinct()
