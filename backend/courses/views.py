"""Views for the Courses app with enhanced filtering, ordering, search, and pagination."""

from rest_framework import filters, viewsets
from django_filters.rest_framework import DjangoFilterBackend
from prism_backend.mixins import CachedViewMixin

from .models import (
    CourseCatalog,
    CourseInstances,
    Semester,
    CourseAssignmentCollaboration,
    Students,
    StudentEnrollments,
    Professors,
    ProfessorEnrollments,
    TeachingAssistants,
    TeachingAssistantEnrollment,
)
from .serializers import (
    CourseCatalogSerializer,
    CourseInstancesSerializer,
    SemesterSerializer,
    CourseAssignmentCollaborationSerializer,
    StudentsSerializer,
    StudentEnrollmentsSerializer,
    ProfessorsSerializer,
    ProfessorEnrollmentsSerializer,
    TeachingAssistantsSerializer,
    TeachingAssistantEnrollmentSerializer,
)
from .pagination import StandardResultsSetPagination


class CourseCatalogViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling CourseCatalog entries."""

    queryset = CourseCatalog.objects.all()
    serializer_class = CourseCatalogSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["subject", "catalog_number", "course_title"]
    ordering_fields = ["catalog_number", "course_title"]
    ordering = ["catalog_number"]
    search_fields = ["name", "course_title"]


class CourseInstancesViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling CourseInstances entries."""

    queryset = CourseInstances.objects.all()
    serializer_class = CourseInstancesSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["section_number", "canvas_course_id"]
    ordering_fields = ["section_number"]
    ordering = ["section_number"]
    search_fields = ["course_catalog__course_title"]


class SemesterViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling Semester entries."""

    queryset = Semester.objects.all()
    serializer_class = SemesterSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["year", "term"]
    ordering_fields = ["year"]
    ordering = ["year"]
    search_fields = ["name", "term", "session"]


class CourseAssignmentCollaborationViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling CourseAssignmentCollaboration entries."""

    queryset = CourseAssignmentCollaboration.objects.all()
    serializer_class = CourseAssignmentCollaborationSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["assignment", "course_instance"]
    ordering_fields = ["assignment", "course_instance"]
    ordering = ["assignment"]
    search_fields = []


class StudentsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling Students entries."""

    queryset = Students.objects.all()
    serializer_class = StudentsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["email", "nshe_id"]
    ordering_fields = ["first_name", "last_name"]
    ordering = ["first_name"]
    search_fields = ["first_name", "last_name", "ace_id"]


class StudentEnrollmentsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling StudentEnrollments entries."""

    queryset = StudentEnrollments.objects.all()
    serializer_class = StudentEnrollmentsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["student", "course_instance"]
    ordering_fields = ["student"]
    ordering = ["student"]
    search_fields = []


class ProfessorsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling Professors entries."""

    queryset = Professors.objects.all()
    serializer_class = ProfessorsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["user"]
    ordering_fields = ["user"]
    ordering = ["user"]
    search_fields = ["user__username"]


class ProfessorEnrollmentsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling ProfessorEnrollments entries."""

    queryset = ProfessorEnrollments.objects.all()
    serializer_class = ProfessorEnrollmentsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["professor", "course_instance"]
    ordering_fields = ["professor"]
    ordering = ["professor"]
    search_fields = []


class TeachingAssistantsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling TeachingAssistants entries."""

    queryset = TeachingAssistants.objects.all()
    serializer_class = TeachingAssistantsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["user"]
    ordering_fields = ["user"]
    ordering = ["user"]
    search_fields = ["user__username"]


class TeachingAssistantEnrollmentViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling TeachingAssistantEnrollment entries."""

    queryset = TeachingAssistantEnrollment.objects.all()
    serializer_class = TeachingAssistantEnrollmentSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = ["teaching_assistant", "course_instance"]
    ordering_fields = ["teaching_assistant"]
    ordering = ["teaching_assistant"]
    search_fields = []
