"""Assignment views with enhanced filtering, ordering, and search capabilities."""

import pandas as pd
from rest_framework import filters, viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from prism_backend.mixins import CachedViewMixin
from users.permissions import IsProfessorOrAdmin
from django.db.models import Max, Avg, F
from assignments.models import SubmissionSimiliarityPairs #include this in the .models down below?

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


class AggregatedAssignmentDataView(APIView):
    """
    API endpoint for aggregated assignment data.
    
    Returns combined statistics (i.e. average grade) for submissions
    grouped by student and assignment. Professors, TAs, and admins can
    filter by students, assignments, and course.
    """
    
    # just in case we add other perms...
    permission_classes = [IsAuthenticated, IsProfessorOrAdmin]

    def get(self, request, format=None):
        student_ids = request.query_params.getlist("students")
        assignment_ids = request.query_params.getlist("assignments")
        course_id = request.query_params.get("course")

        '''qs = Submissions.objects.select_related(
            "student", "assignment", "course_instance", "course_instance__professor"
        ).all()'''

        # Build a queryset that retrieves data along with similarity scores.
        qs = SubmissionSimiliarityPairs.objects.select_related(
            "assignment",
            "submission_id_1__student",
            "submission_id_2"  # add other stuff if we want to show different things
        ).values(
            "assignment__title",                           
            "assignment__assignment_number",               
            "submission_id_1__student__first_name",          
            "submission_id_1__student__last_name",           
            "percentage"    # Similarity score
            # other fields here later...
        )

        if student_ids:
            qs = qs.filter(submission_id_1__student__id__in=student_ids)
        if assignment_ids:
            qs = qs.filter(assignment__id__in=assignment_ids)
        if course_id:
            qs = qs.filter(course_instance_id=course_id)


        #didnt want to use the dataframe method for this. Stuck to ORM aggregations
        # qs becomes a list of dicts
        '''data = list(qs)
        if not data:
            return Response({"message": "No data found for the specified filters."}, status=status.HTTP_404_NOT_FOUND)

        df = pd.DataFrame.from_records(data)
        # TODO: Check this is actually a date
        df["created_at"] = pd.to_datetime(df["created_at"])'''


        # EVERYTHING gonna go in here ;)
        response_data = {}

        '''Theres basically 2 strategies to filling response_data. 
        Doing aggregations with .objects.values(...).annotate(...) (aka ORM aggregations)
        keeps aggregations in the db, meaning its faster bc of less data transferred on network. It only returns aggregated fields
        instead of full model instnaces or all raw data, this makes it fast too. Its simple for standard aggregations like
        average and max. 
        Disadvantages: More complex aggregations are a LOT harder. It can also be annoying to do multiple aggregations in the db.
        '''

        # Highest sim score per student
        response_data["student_max_similarity_score"] = list(SubmissionSimiliarityPairs.objects.values(
            "submission_id_1__student__id",
            "submission_id_1__student__first_name",
            "submission_id_1__student__last_name",
        ).annotate(
            max_score=Max("percentage")
        ))
        
        # Average sim score per assignment
        response_data["assignment_avg_similarity_score"] = list(SubmissionSimiliarityPairs.objects.values(
            "assignment__id",
            "assignment__title"
        ).annotate(
            avg_score=Avg("percentage")
        ))
        
        # Flagged submissions per assignment
        #the 'F' lets each sim score be compared against its assignments own threshold. (dynamic)
        response_data["flagged_per_assignment"] = list(SubmissionSimiliarityPairs.objects.filter(
            percentage__gte=F('assignment__requiredsubmissionfile_set__similarity_threshold')
        ).values(
            "assignment__title"
        ).annotate(
            flagged_count=Count("id")
        ))
    
        # similarity score over time trend
        response_data["similarity_trends"] = list(
            Submissions.objects.values("created_at")
            .annotate(avg_similarity=Avg("submission__submissionsimiliaritypairs__percentage"))
        )

        # Flagged submissions per professor
        response_data["flagged_by_professor"] = list(SubmissionSimiliarityPairs.objects.filter(
            percentage__gte=F('assignment__requiredsubmissionfiles__similarity_threshold'))
            .values("assignment__course_instance__professor__user__username")
            .annotate(flagged_count=Count("id"))
        )

        # Professor-wise average similarity score. We probably want admins to see this for all professors
        response_data["professor_avg_similarity"] = list(
            SubmissionSimiliarityPairs.objects.values("assignment__course_instance__professor__user__username")
            .annotate(avg_similarity=Avg("percentage"))
        )
        
        # return Response(avgGradeGroup.to_dict(orient="records"), status=status.HTTP_200_OK)
        return Response(response_data, status=status.HTTP_200_OK)
