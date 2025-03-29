"""Courses Views with Enhanced Filtering, Ordering, and Search Capabilities.

This module defines viewsets for managing course-related models, including
Professor, Semester, Class, ProfessorClassSection, and Enrollment. Each viewset
provides CRUD operations along with filtering, ordering, and searching
capabilities to facilitate efficient data retrieval and manipulation.

ViewSets:
    - ProfessorVS: Manages Professor objects.
    - SemesterVS: Manages Semester objects.
    - ClassVS: Manages Class objects.
    - ProfessorClassSectionVS: Manages ProfessorClassSection objects.
    - EnrollmentVS: Manages Enrollment objects.

Each viewset utilizes Django REST Framework's ModelViewSet and integrates
filtering, ordering, and searching functionalities through appropriate
backends and field configurations.
"""

from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.filters import OrderingFilter, SearchFilter
from django_filters import rest_framework as filters
from courses import serializers
from courses import models


class ProfessorVS(viewsets.ModelViewSet):
    """ViewSet for managing Professor objects.

    Provides CRUD operations for the Professor model and supports filtering,
    ordering, and searching based on specific fields.

    Attributes:
        queryset (QuerySet): All Professor objects.
        serializer_class (Serializer): Serializer for Professor objects.
        filter_backends (list): Backends for filtering, ordering, and searching.
        filterset_fields (dict): Fields available for filtering.
        ordering_fields (list): Fields available for ordering results.
        ordering (list): Default ordering for the queryset.
        search_fields (list): Fields available for search queries.
    """

    queryset = models.Professor.objects.all()
    serializer_class = serializers.ProfessorSerializer
    filter_backends = [
        filters.DjangoFilterBackend,
        OrderingFilter,
        SearchFilter,
    ]
    filterset_fields = {
        "id": ["exact"],
        "user__first_name": ["exact", "icontains"],
    }
    ordering_fields = ["user__first_name"]
    ordering = ["user__first_name"]
    search_fields = ["user__first_name"]


class SemesterVS(viewsets.ModelViewSet):
    """ViewSet for managing Semester objects.

    Provides CRUD operations for the Semester model, along with filtering,
    ordering, and searching capabilities.

    Attributes:
        queryset (QuerySet): All Semester objects.
        serializer_class (Serializer): Serializer for Semester objects.
        filter_backends (list): Backends for filtering, ordering, and searching.
        filterset_fields (dict): Fields available for filtering.
        ordering_fields (list): Fields available for ordering results.
        ordering (list): Default ordering for the queryset.
        search_fields (list): Fields available for search queries.
    """

    queryset = models.Semester.objects.all()
    serializer_class = serializers.SemesterSerializer
    filter_backends = [
        filters.DjangoFilterBackend,
        OrderingFilter,
        SearchFilter,
    ]
    filterset_fields = {"id": ["exact"], "name": ["exact", "icontains"]}
    ordering_fields = ["name"]
    ordering = ["name"]
    search_fields = ["name"]


class ClassVS(viewsets.ModelViewSet):
    """ViewSet for managing Class objects.

    Provides CRUD operations for the Class model, along with filtering,
    ordering, and searching capabilities.

    Attributes:
        queryset (QuerySet): All Class objects.
        serializer_class (Serializer): Serializer for Class objects.
        filter_backends (list): Backends for filtering, ordering, and searching.
        filterset_fields (dict): Fields available for filtering.
        ordering_fields (list): Fields available for ordering results.
        ordering (list): Default ordering for the queryset.
        search_fields (list): Fields available for search queries.
    """

    queryset = models.Class.objects.all()
    serializer_class = serializers.ClassSerializer
    filter_backends = [
        filters.DjangoFilterBackend,
        OrderingFilter,
        SearchFilter,
    ]
    filterset_fields = {"id": ["exact"], "name": ["exact", "icontains"]}
    ordering_fields = ["name"]
    ordering = ["name"]
    search_fields = ["name"]


class ProfessorClassSectionVS(viewsets.ModelViewSet):
    """ViewSet for managing ProfessorClassSection objects.

    Provides CRUD operations for the ProfessorClassSection model and supports
    filtering, ordering, and searching on related fields.

    Attributes:
        queryset (QuerySet): All ProfessorClassSection objects.
        serializer_class (Serializer): Serializer for ProfessorClassSection objects.
        filter_backends (list): Backends for filtering, ordering, and searching.
        filterset_fields (dict): Fields available for filtering, including related fields.
        ordering_fields (list): Fields available for ordering results.
        ordering (list): Default ordering for the queryset.
        search_fields (list): Fields available for search queries.
    """

    queryset = models.ProfessorClassSection.objects.all()
    serializer_class = serializers.ProfessorClassSectionSerializer
    filter_backends = [
        filters.DjangoFilterBackend,
        OrderingFilter,
        SearchFilter,
    ]
    filterset_fields = {
        "professor__id": ["exact"],
        "class_instance__name": ["exact", "icontains"],
        "semester__name": ["exact", "icontains"],
    }
    ordering_fields = [
        "professor__user__first_name",
        "class_instance__name",
        "semester__name",
    ]
    ordering = ["semester__name", "class_instance__name"]
    search_fields = ["class_instance__name", "semester__name"]


class EnrollmentVS(viewsets.ModelViewSet):
    """ViewSet for managing Enrollment objects.

    Provides CRUD operations for the Enrollment model and supports filtering,
    ordering, and searching functionalities.

    Attributes:
        queryset (QuerySet): All Enrollment objects.
        serializer_class (Serializer): Serializer for Enrollment objects.
        permission_classes (list): Permissions required for access.
        filter_backends (list): Backends for filtering, ordering, and searching.
        filterset_fields (dict): Fields available for filtering.
        ordering_fields (list): Fields available for ordering results.
        ordering (list): Default ordering for the queryset.
        search_fields (list): Fields available for search queries.
    """

    queryset = models.Enrollment.objects.all()
    serializer_class = serializers.EnrollmentSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [
        filters.DjangoFilterBackend,
        OrderingFilter,
        SearchFilter,
    ]
    filterset_fields = {
        "student__id": ["exact"],
        "class_instance__id": ["exact"],
        "semester__id": ["exact"],
    }
    ordering_fields = ["enrolled_date", "semester__name"]
    ordering = ["enrolled_date"]
    search_fields = [
        "student__user__first_name",
        "class_instance__name",
        "semester__name",
    ]
