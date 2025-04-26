"""Views for the Cheating app with enhanced filtering, ordering, search, and pagination."""

from rest_framework import filters, viewsets
from django_filters.rest_framework import DjangoFilterBackend
from prism_backend.mixins import CachedViewMixin

from .models import (
    CheatingGroups,
    CheatingGroupMembers,
    ConfirmedCheaters,
    FlaggedStudents,
    SubmissionSimilarityPairs,
    LongitudinalCheatingGroups,
    LongitudinalCheatingGroupMembers,
    LongitudinalCheatingGroupInstances,
)
from .serializers import (
    CheatingGroupsSerializer,
    CheatingGroupMembersSerializer,
    ConfirmedCheatersSerializer,
    FlaggedStudentsSerializer,
    SubmissionSimilarityPairsSerializer,
    LongitudinalCheatingGroupsSerializer,
    LongitudinalCheatingGroupMembersSerializer,
    LongitudinalCheatingGroupInstancesSerializer,
)
from .pagination import StandardResultsSetPagination

# for the pdf / csv export
import io
import csv
from datetime import datetime

from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from users.permissions import IsProfessorOrAdmin
# from cheating.models import SubmissionSimiliarityPairs
from courses.models import CourseInstances
from assignments.models import Assignments


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


class SubmissionSimilarityPairsViewSet(viewsets.ModelViewSet, CachedViewMixin):
    """ViewSet for handling SubmissionSimilarityPairs entries."""

    queryset = SubmissionSimilarityPairs.objects.all()
    serializer_class = SubmissionSimilarityPairsSerializer
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


class ExportPlagiarismReportView(APIView):
    """
    Export plagiarism/similarity data as PDF or CSV.

    GET params:
      - ?course_instance=<id>
      - ?assignment=<id>      (optional: limit to one assignment)
      - ?format=pdf|csv       (defaults to csv)
    """
    permission_classes = [IsAuthenticated, IsProfessorOrAdmin]

    def get(self, request, format=None):
        # get filters
        ci_id = request.query_params.get("course_instance")
        asg_id = request.query_params.get("assignment")
        #TODO: should format be pdf?
        output_format = request.query_params.get("format", "csv").lower()

        qs = SubmissionSimiliarityPairs.objects.select_related(
            "assignment",
            "submission_id_1__student",
            "submission_id_2__student",
        ).all()

        if ci_id:
            qs = qs.filter(assignment__course_instance_id=ci_id)
        if asg_id:
            qs = qs.filter(assignment_id=asg_id)

        # build a list of rows (all the data we want to print out)
        rows = []
        for sim in qs:
            rows.append([
                sim.assignment.course_instance,  # i.e. “CS 472 - Section 1001”
                sim.assignment.assignment_number,
                sim.assignment.title,
                f"{sim.submission_id_1.student.first_name} {sim.submission_id_1.student.last_name}",
                f"{sim.submission_id_2.student.first_name} {sim.submission_id_2.student.last_name}",
                sim.file_name,
                f"{sim.percentage}%",
            ])

        if output_format == "pdf":
            return self._export_pdf(rows, ci_id, asg_id)
        else:
            return self._export_csv(rows, ci_id, asg_id)

    
    def _export_csv(self, rows, ci_id, asg_id):
        buf = io.StringIO()
        writer = csv.writer(buf)
        # header info. Its the first row of the writer output
        writer.writerow([
            "Course Instance",
            "Assignment #",
            "Assignment Title",
            "Student A",
            "Student B",
            "File Name",
            "Similarity %"
        ])
        writer.writerows(rows)

        resp = HttpResponse(buf.getvalue(), content_type="text/csv")
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        # content-disposition and attatchment should tell the browser to download the file
        resp["Content-Disposition"] = f'attachment; filename="plagiarism_{ts}.csv"'
        return resp

    def _export_pdf(self, rows, ci_id, asg_id):
        buf = io.BytesIO()
        # doc and styles uses ReportLab
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()

        # Title stuff. append specific course and asg info if its available to us
        title_text = "Plagiarism Report"
        if ci_id:
            ci = CourseInstances.objects.get(id=ci_id)
            title_text += f" — {ci}"
        if asg_id:
            asg = Assignments.objects.get(id=asg_id)
            title_text += f" / Assignment {asg.assignment_number}"
        # Paragraph is also from ReportLab
        story = [Paragraph(title_text, styles["Title"]), Spacer(1, 12)]

        # This is the header data, followed by the data (rows)
        data = [[
            "Course Instance",
            "Assign #",
            "Title",
            "Student A",
            "Student B",
            "File",
            "Similarity"
        ]] + rows


        #finish thiss
        # add a table varaible, and do .setStyle stuff.
        # do store.append(table)?
        # doc.build(story)

        resp = HttpResponse(buf.getvalue(), content_type="application/pdf")
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        resp["Content-Disposition"] = f'attachment; filename="plagiarism_{ts}.pdf"'
        return resp
