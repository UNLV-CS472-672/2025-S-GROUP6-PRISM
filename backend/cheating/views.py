"""Views for the Cheating app with enhanced filtering, ordering, search, and pagination."""

from django.db import connection
import io
import math
from matplotlib.patches import Patch
import scipy.stats as stats
import matplotlib.patheffects as pe
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from courses.models import CourseCatalog, Semester
import matplotlib
from matplotlib import pyplot as plt
import time
import numpy as np
from django.db import transaction
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import filters, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from django_filters.rest_framework import DjangoFilterBackend
from prism_backend.mixins import CachedViewMixin
from assignments.models import Assignments, Submissions
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA
from matplotlib.patches import Circle
from .services import (
    bulk_recompute_semester_profiles,
    run_kmeans_for_course_semester,
)
from .utils.data_analysis import (
    populate_student_pair_stats,
    run_pair_level_clustering,
)
from .services import generate_report as generate_report_service
from django.views.decorators.http import require_GET

from courses.models import Students
from .models import (
    CheatingGroups,
    CheatingGroupMembers,
    ConfirmedCheaters,
    FlaggedStudentPair,
    FlaggedStudents,
    PairFlagStat,
    SubmissionSimilarityPairs,
    LongitudinalCheatingGroups,
    LongitudinalCheatingGroupMembers,
    LongitudinalCheatingGroupInstances,
    AssignmentReport,
    StudentReport,
    StudentSemesterProfile,
    StudentPairSimilarityStatistics,
)

from courses.serializers import StudentsSerializer
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

from .utils.similarity_analysis import (
    get_all_scores_by_student,
    update_all_pair_stats,
)

# NOTE: The following import is commented out because it is not used in this file.
# from django.views.decorators.http import require_POST
# we may need this later because some of the views create new ojects
# in the database and we may want to use POST instead of GET for those views.
# right now all is get for easy of use without account creation

# Set up non-interactive backend for matplotlib
matplotlib.use("Agg")


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

    queryset = SubmissionSimilarityPairs.objects.select_related(
        "submission_id_1", "submission_id_2", "assignment"
    )
    serializer_class = SubmissionSimilarityPairsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.OrderingFilter,
        filters.SearchFilter,
    ]
    filterset_fields = [
        "assignment",
        "file_name",
        "match_id",
        "submission_id_1__student_id",
        "submission_id_2__student_id",
        "submission_id_1__assignment_id",
        "submission_id_1__assignment__semester_id",
        "submission_id_1__course_instance",
        "submission_id_2__course_instance",
    ]
    ordering_fields = ["percentage", "match_id", "file_name"]
    ordering = ["percentage"]
    search_fields = [
        "file_name",
        "assignment__title",
        "submission_id_1__student__first_name",
        "submission_id_2__student__first_name",
    ]

    def get_queryset(self):
        """
        Return a filtered queryset of submission similarity pairs.

        Query parameters:
        - student1: ID of the first student
        - student2: ID of the second student
        - asid: Assignment ID
        - course: CourseInstance ID (must match both submissions)
        - course1 + course2: Allow either submission to match
        - semester: Semester ID (must match both submissions)
        - semester1 + semester2: Allow either submission to match
        """
        queryset = SubmissionSimilarityPairs.objects.select_related(
            "assignment",
            "submission_id_1__student",
            "submission_id_1__course_instance",
            "submission_id_1__assignment__semester",
            "submission_id_2__student",
            "submission_id_2__course_instance",
            "submission_id_2__assignment__semester",
        )

        request = self.request
        student1 = request.query_params.get("student1")
        student2 = request.query_params.get("student2")
        assignment_id = request.query_params.get("asid")
        course = request.query_params.get("course")
        course1 = request.query_params.get("course1")
        course2 = request.query_params.get("course2")
        semester = request.query_params.get("semester")
        semester1 = request.query_params.get("semester1")
        semester2 = request.query_params.get("semester2")

        if student1:
            queryset = queryset.filter(submission_id_1__student_id=student1)
        if student2:
            queryset = queryset.filter(submission_id_2__student_id=student2)
        if assignment_id:
            queryset = queryset.filter(
                submission_id_1__assignment_id=assignment_id,
                submission_id_2__assignment_id=assignment_id,
                assignment_id=assignment_id,
            )

        # Course filtering
        if course:
            queryset = queryset.filter(
                submission_id_1__course_instance_id=course,
                submission_id_2__course_instance_id=course,
            )
        elif course1 and course2:
            queryset = queryset.filter(
                submission_id_1__course_instance_id=course1,
                submission_id_2__course_instance_id=course2,
            )

        # Semester filtering
        if semester:
            queryset = queryset.filter(
                submission_id_1__assignment__semester_id=semester,
                submission_id_2__assignment__semester_id=semester,
            )
        elif semester1 and semester2:
            queryset = queryset.filter(
                submission_id_1__assignment__semester_id=semester1,
                submission_id_2__assignment__semester_id=semester2,
            )

        return queryset

    @action(detail=False, methods=["get"], url_path="students-with-similarities")
    def students_with_similarities(self, request):
        """Return all students involved in at least one similarity pair."""
        student_ids_1 = self.get_queryset().values_list(
            "submission_id_1__student_id", flat=True
        )
        student_ids_2 = self.get_queryset().values_list(
            "submission_id_2__student_id", flat=True
        )

        combined_ids = student_ids_1.union(student_ids_2)

        students = (
            Students.objects.filter(id__in=combined_ids)
            .distinct()
            .order_by("last_name", "first_name")
        )

        page = self.paginate_queryset(students)
        if page is not None:
            serializer = StudentsSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = StudentsSerializer(students, many=True)
        return Response(serializer.data)


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


@require_GET
def similarity_plot(request, assignment_id):
    """
    Render a vertical bar chart of students whose z-score exceeds our cutoff.

    We:
    1. Load the Assignment and its latest AssignmentReport in just two queries.
    2. Fetch all StudentReport rows with z_score > cutoff (2.0 for the 95th percentile).
    3. Bulk-load student names via in_bulk to avoid N+1 queries.
    4. Pair each name with its z-score, sort descending, and plot a bar chart.
    """
    # Step 1: Fetch assignment and its report
    assignment = get_object_or_404(Assignments, pk=assignment_id)
    report = get_object_or_404(AssignmentReport, assignment=assignment)

    # Step 2: Choose z-score cutoff—2.0 corresponds roughly to the 95% tail
    cutoff = 2.0

    # Step 3: Query only the fields we need to minimize data transfer
    flagged_qs = StudentReport.objects.filter(report=report, z_score__gt=cutoff).only(
        "submission_id", "z_score"
    )

    if not flagged_qs.exists():
        raise Http404("No flagged students")

    # Step 4: Bulk-fetch the related Submissions to retrieve student names
    sub_ids = [sr.submission_id for sr in flagged_qs]
    sub_map = (
        Submissions.objects.filter(pk__in=sub_ids)
        .select_related("student")
        .in_bulk(field_name="pk")
    )

    # Step 5: Build a list of (student name, z-score)
    entries = []
    for sr in flagged_qs:
        submission = sub_map.get(sr.submission_id)
        if submission:
            name = f"{submission.student.first_name} " f"{submission.student.last_name}"
        else:
            name = f"ID {sr.submission_id}"
        entries.append((name, sr.z_score))

    # Step 6: Sort by z-score descending for the bar chart
    entries.sort(key=lambda x: x[1], reverse=True)
    names, zs = zip(*entries)

    # Step 7: Create the bar chart, sizing width by number of bars
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    ax.bar(range(len(names)), zs, color="C1", edgecolor="black")

    # Step 8: Annotate each bar with its z-score
    for i, z in enumerate(zs):
        ax.text(i, z + 0.02, f"{z:.2f}", ha="center", va="bottom", fontsize=8)

    # Step 9: Draw a horizontal line at the cutoff
    ax.axhline(cutoff, color="red", linestyle="--", label=f"Cutoff: z > {cutoff:.2f}")

    # Step 10: Label and rotate x-ticks for readability
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Z-score of Mean Similarity")
    ax.set_title(f"Assignment {assignment_id}: Flagged Students")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    # Step 11: Stream the plot back as a PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.read(), content_type="image/png")


@require_GET
def distribution_plot(request, assignment_id):
    """
    Render a histogram of per-student mean similarities with a Normal PDF overlay.

    We:
    1. Load all mean_similarity values from StudentReport in one query.
    2. Compute the SE for the Normal PDF using CLT on average sample size.
    3. Plot the histogram and overlay the theoretical Normal curve.
    """
    # Load assignment and its stats report
    assignment = get_object_or_404(Assignments, pk=assignment_id)
    report = get_object_or_404(AssignmentReport, assignment=assignment)

    # Fetch only submission_id and mean_similarity
    srs = list(
        StudentReport.objects.filter(report=report).only(
            "submission_id", "mean_similarity"
        )
    )

    if not srs:
        raise Http404("No student data to plot")

    means = [sr.mean_similarity for sr in srs]

    # Population stats from the report
    mu = report.mu
    sigma = report.sigma

    # Compute average sample size for SE: reuse cached scores map
    scores_map = get_all_scores_by_student(assignment)
    total_comparisons = sum(len(scores_map.get(sr.submission_id, [])) for sr in srs)
    n_avg = total_comparisons / len(means)
    se = sigma / (n_avg**0.5)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    counts, bins, _ = ax.hist(
        means, bins=30, density=True, alpha=0.6, color="C0", edgecolor="black"
    )

    # Build Normal PDF points
    xs = [bins[0] + i * (bins[-1] - bins[0]) / 200 for i in range(201)]
    pdf = [
        (1 / (se * (2 * math.pi) ** 0.5)) * math.exp(-0.5 * ((x - mu) / se) ** 2)
        for x in xs
    ]
    ax.plot(xs, pdf, "k--", label="Normal PDF (CLT)")

    # Annotate with μ, σ, variance
    stats_text = f"μ = {mu:.2f}\n" f"σ = {sigma:.2f}\n" f"Var = {report.variance:.2f}"
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"),
    )

    ax.set_xlabel("Per-student Mean Similarity (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"Assignment {assignment_id}: Distribution")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    # Stream PNG back
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.read(), content_type="image/png")


@require_GET
def similarity_interval_plot(request, assignment_id):
    """
    Render a forest plot of 95% confidence intervals for flagged students.

    We:
    1. Select only StudentReports with z_score > cutoff (2.0).
    2. Bulk-fetch names to avoid N+1 queries.
    3. Plot mean ± CI as horizontal error bars, with names on the y-axis.
    """
    # Fetch assignment and its report
    assignment = get_object_or_404(Assignments, pk=assignment_id)
    report = get_object_or_404(AssignmentReport, assignment=assignment)

    # z-score threshold
    cutoff = 2.0
    srs = list(
        StudentReport.objects.filter(report=report, z_score__gt=cutoff).only(
            "submission_id", "mean_similarity", "ci_lower", "ci_upper", "z_score"
        )
    )

    if not srs:
        raise Http404("No flagged students to plot")

    # Bulk-fetch student names
    sub_ids = [sr.submission_id for sr in srs]
    sub_map = (
        Submissions.objects.filter(pk__in=sub_ids)
        .select_related("student")
        .in_bulk(field_name="pk")
    )

    # Build display data
    data = []
    for sr in srs:
        submission = sub_map.get(sr.submission_id)
        if submission:
            name = f"{submission.student.first_name} " f"{submission.student.last_name}"
        else:
            name = f"ID {sr.submission_id}"
        data.append((name, sr.mean_similarity, sr.ci_lower, sr.ci_upper, sr.z_score))

    # Sort by mean similarity descending
    data.sort(key=lambda x: x[1], reverse=True)
    names, means, lowers, uppers, zs = zip(*data)

    # Compute error bar widths
    left_err = [m - lo for m, lo in zip(means, lowers)]
    right_err = [hi - m for hi, m in zip(uppers, means)]

    # Plot forest plot
    fig, ax = plt.subplots(figsize=(6, max(4, len(names) * 0.4)))
    ax.errorbar(
        means,
        list(range(len(names))),
        xerr=[left_err, right_err],
        fmt="o",
        capsize=4,
        color="C1",
    )

    # Reference line at population mean
    ax.axvline(
        report.mu,
        color="grey",
        linestyle="--",
        label=f"Population μ = {report.mu:.1f}%",
    )

    # Annotate with z-scores
    for idx, z in enumerate(zs):
        ax.text(uppers[idx] + 0.5, idx, f"z={z:.2f}", va="center", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Similarity (%) with 95% CI")
    ax.set_title(f"Assignment {assignment_id}: Flagged CI")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    # Stream PNG back
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.read(), content_type="image/png")


@require_GET
def kmeans_clusters_plot(request, course_id, semester_id):
    """
    Render a 2D PCA scatter of student clusters.

    • Pull all StudentSemesterProfile rows (with student names)
      in a single query so we never hit the DB in a loop.
    • Standardize the 7-dim feature vectors and reduce to 2D
      via PCA for visual clarity.
    • Identify low/medium/high risk clusters based on the
      average z-score dimension (feature 0).
    • Draw each cluster as points + hulls or circles, and
      annotate extreme outliers.
    """
    # 1) Fetch all profiles for this course/semester at once.
    profiles = StudentSemesterProfile.objects.filter(
        course_catalog_id=course_id,
        semester_id=semester_id,
    ).select_related("student")
    if not profiles.exists():
        raise Http404("No data for that course and semester.")

    # 2) Extract feature matrix, cluster labels, and student names.
    #    feature_vector is a length-7 list on each profile.
    X_raw = np.vstack([p.feature_vector for p in profiles])
    labels = np.array([p.cluster_label for p in profiles])
    names = [f"{p.student.first_name} {p.student.last_name}" for p in profiles]

    # 3) Standardize features to zero mean and unit variance,
    #    then project down to two principal components.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X_scaled)
    dim1, dim2 = XY[:, 0], XY[:, 1]
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    # 4) Compute each cluster’s centroid on the avg_z (first) feature
    #    without refitting the clusterer.
    unique_labels = sorted(set(labels))
    centroids = {lbl: X_scaled[labels == lbl].mean(axis=0) for lbl in unique_labels}
    avg_z = {lbl: centroids[lbl][0] for lbl in unique_labels}

    #    Determine which label is lowest vs highest risk.
    low_lbl = min(unique_labels, key=lambda label: avg_z[label])
    high_lbl = max(unique_labels, key=lambda label: avg_z[label])

    # 5) Define colors, markers, and legend entries per cluster.
    color_map = {}
    marker_map = {}
    legend_map = {}
    for lbl in unique_labels:
        count = int((labels == lbl).sum())
        if lbl == low_lbl:
            color_map[lbl] = "green"
            marker_map[lbl] = "o"
            legend_map[lbl] = f"Low Intensity (n={count})"
        elif lbl == high_lbl:
            color_map[lbl] = "red"
            marker_map[lbl] = "^"
            legend_map[lbl] = f"High Intensity (n={count})"
        else:
            color_map[lbl] = "gold"
            marker_map[lbl] = "s"
            legend_map[lbl] = f"Medium Intensity (n={count})"

    # 6) Create the figure and plot each cluster group.
    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        xs = dim1[labels == lbl]
        ys = dim2[labels == lbl]
        ax.scatter(
            xs,
            ys,
            c=color_map[lbl],
            marker=marker_map[lbl],
            s=100,
            edgecolor="k",
            alpha=0.8,
            label=legend_map[lbl],
        )

        pts = list(zip(xs, ys))
        # Draw a convex hull if ≥3 points
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = [pts[i] for i in hull.vertices]
                hx, hy = zip(*hull_pts)
                ax.fill(hx, hy, color=color_map[lbl], alpha=0.2)
            except QhullError:
                pass
        # Draw a circle for 1–2 points
        elif pts:
            cx, cy = np.mean(xs), np.mean(ys)
            radius = (np.hypot(xs - cx, ys - cy).max() if len(xs) > 1 else 0.5) * 1.5
            circ = Circle((cx, cy), radius=radius, color=color_map[lbl], alpha=0.2)
            ax.add_patch(circ)

    # 7) Annotate extreme outliers beyond the 10th/90th percentiles.
    x_lo, x_hi = np.percentile(dim1, [10, 90])
    y_lo, y_hi = np.percentile(dim2, [10, 90])
    for x, y, name in zip(dim1, dim2, names):
        if not (x_lo < x < x_hi and y_lo < y < y_hi):
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.9,
            )

    # 8) Final styling: labels, title, grid, legend.
    course = get_object_or_404(CourseCatalog, pk=course_id)
    semester = get_object_or_404(Semester, pk=semester_id)

    ax.set_xlabel(f"Dim1 ({var1:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"Dim2 ({var2:.1f}% var)", fontsize=12)
    ax.set_title(f"K-Means Clusters → {course} — {semester}", fontsize=14, pad=15)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Student Groups")
    plt.tight_layout()

    # 9) Output the figure as a PNG image.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.read(), content_type="image/png")


def run_kmeans_for_pair_stats(
    course_id,
    semester_id,
    student_km=None,
    n_clusters=6,
    random_state=0,
    b_refs=10,
    red_threshold=30,
    max_multiplier=2,
):
    """
    1) Optionally use student_km for per-student labels.

    2) Build an 8-dim feature matrix for each pair:
       [ mean_z*|mean_z|,
         max_z*|max_z|,
         mean_similarity,
         proportion*100,
         sum_of_student_labels,
         flagged_count²,
         total_similarity,
         total_z_score ]
    3) Standardize + gap statistic → pick best_k.
    4) Fit KMeans(best_k) → find “red” cluster;
       if red_size > red_threshold: clear flags,
       else bulk-create new FlaggedStudentPair rows.
    5) Remap PairFlagStat.kmeans_label so 0 = lowest risk.
    """
    start_ts = time.time()

    # 1) load all pair stats at once
    pairs = list(
        PairFlagStat.objects.filter(
            course_catalog_id=course_id,
            semester_id=semester_id,
        )
    )
    if not pairs:
        raise Http404("No pair statistics available")

    # 1a) build student_id → cluster_label map
    if student_km:
        label_map = dict(zip(student_km._fit_student_ids_, student_km.labels_))
    else:
        label_map = {
            p.student_id: (p.cluster_label or 0)
            for p in StudentSemesterProfile.objects.filter(
                course_catalog_id=course_id,
                semester_id=semester_id,
            ).only("student_id", "cluster_label")
        }

    # 2) assemble raw 8-dim matrix
    X_raw = np.zeros((len(pairs), 8), dtype=float)
    for i, ps in enumerate(pairs):
        # squared z-scores preserve sign and accentuate large values
        X_raw[i, 0] = ps.mean_z_score * abs(ps.mean_z_score)
        X_raw[i, 1] = ps.max_z_score * abs(ps.max_z_score)
        # raw average similarity and percent of comparisons
        X_raw[i, 2] = ps.mean_similarity
        X_raw[i, 3] = ps.proportion * 100.0
        # link to student-level risk clusters
        X_raw[i, 4] = label_map.get(ps.student_a_id, 0) + label_map.get(
            ps.student_b_id, 0
        )
        # cumulative indicators from PairFlagStat
        X_raw[i, 5] = ps.flagged_count**2
        X_raw[i, 6] = ps.total_similarity
        X_raw[i, 7] = ps.total_z_score

    # 3) standardize so each feature has mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    def _dispersion(data, labels, centers):
        """Compute within-cluster dispersion for gap statistic."""
        return sum(
            np.sum((data[labels == lbl] - ctr) ** 2) for lbl, ctr in enumerate(centers)
        )

    # 3a) gap statistic to select best_k
    mins = X_scaled.min(axis=0)
    maxs = X_scaled.max(axis=0)
    best_gap = -np.inf
    best_k = 2
    for k in range(2, n_clusters + 1):
        km_ref = KMeans(n_clusters=k, random_state=random_state).fit(X_scaled)
        Wk = _dispersion(X_scaled, km_ref.labels_, km_ref.cluster_centers_)
        logWk = np.log(Wk)
        ref_logs = []
        for _ in range(b_refs):
            Xb = np.random.uniform(mins, maxs, X_scaled.shape)
            kmb = KMeans(n_clusters=k, random_state=random_state).fit(Xb)
            Wkb = _dispersion(Xb, kmb.labels_, kmb.cluster_centers_)
            ref_logs.append(np.log(Wkb))
        gap_k = np.mean(ref_logs) - logWk
        if gap_k > best_gap:
            best_gap, best_k = gap_k, k

    # 4) final KMeans fit and red-cluster logic
    km_final = KMeans(n_clusters=best_k, random_state=random_state)
    labels = km_final.fit_predict(X_scaled)

    # recover raw centroids to score clusters
    centers_raw = scaler.inverse_transform(km_final.cluster_centers_)
    composites = centers_raw[:, 0] + centers_raw[:, 1] + centers_raw[:, 3]
    red_label = int(np.argmax(composites))
    red_size = int(np.bincount(labels, minlength=best_k)[red_label])

    # clear any old flagged pairs first
    FlaggedStudentPair.objects.filter(
        course_catalog_id=course_id,
        semester_id=semester_id,
    ).delete()

    # if red cluster is small enough, flag each member
    if red_size <= red_threshold:
        flagged = []
        for ps, lbl in zip(pairs, labels):
            if lbl == red_label:
                max_sim = getattr(ps, "max_similarity", ps.mean_similarity)
                flagged.append(
                    FlaggedStudentPair(
                        course_catalog_id=course_id,
                        semester_id=semester_id,
                        student_a_id=ps.student_a_id,
                        student_b_id=ps.student_b_id,
                        mean_similarity=ps.mean_similarity,
                        max_similarity=max_sim,
                        mean_z_score=ps.mean_z_score,
                        max_z_score=ps.max_z_score,
                    )
                )
        if flagged:
            with transaction.atomic():
                FlaggedStudentPair.objects.bulk_create(flagged, ignore_conflicts=True)

    # 5) remap labels so 0 = lowest-risk cluster
    order = np.argsort(composites)
    remap = {old: new for new, old in enumerate(order)}
    for ps, lbl in zip(pairs, labels):
        ps.kmeans_label = remap[int(lbl)]
    with transaction.atomic():
        PairFlagStat.objects.bulk_update(pairs, ["kmeans_label"])

    print(
        f"✔️ run_kmeans_for_pair_stats done in "
        f"{time.time() - start_ts:.2f}s; red_size={red_size}"
    )
    return km_final


def _generate_one(assignment_id):
    """
    Help for ThreadPoolExecutor: close the old DB connection.

    then run the report service in a fresh one.
    """
    connection.close()
    return generate_report_service(assignment_id)


@require_GET
def run_full_pipeline(request, course_id: int, semester_id: int):
    """
    0) Clear old AssignmentReport rows.

    1) Clear old PairFlagStat rows.
    2) Generate each assignment report in parallel.
    3) Bulk-upsert flagged-pair stats.
    4) Clear & rebuild StudentSemesterProfile.
    5) Run student-level K-Means.
    6) Run data-analytics pipeline (per-pair stats + pair-level K-Means).
    """
    start_ts = time.time()
    print(
        f"\n🚀 [PIPELINE] Starting full pipeline for course={course_id}, semester={semester_id}"
    )

    # 0) Drop old assignment reports
    AssignmentReport.objects.filter(
        assignment__course_catalog_id=course_id,
        assignment__semester_id=semester_id,
    ).delete()
    print("  🔄 Cleared old AssignmentReport rows")

    # 1) Drop old pair stats
    PairFlagStat.objects.filter(
        course_catalog_id=course_id,
        semester_id=semester_id,
    ).delete()
    print("  🔄 Cleared old PairFlagStat rows")

    # 2) Parallel report generation
    assignments = list(
        Assignments.objects.filter(
            course_catalog_id=course_id,
            semester_id=semester_id,
        )
    )
    if not assignments:
        raise Http404("No assignments for that course+semester.")
    print(f"  📋 Found {len(assignments)} assignments; generating reports…")

    all_flagged = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_generate_one, a.id): a for a in assignments}
        for fut in as_completed(futures):
            a = futures[fut]
            try:
                report, flagged = fut.result()
            except Exception as exc:
                print(f"    ❌ generate_report({a.id}) failed:", exc)
                continue
            if report:
                print(f"    ▶ report.id={report.id}, flagged_pairs={len(flagged)}")
                all_flagged.extend(flagged)

    # 3) Bulk-upsert all flagged pairs
    print(f"  🔄 Bulk-upserting {len(all_flagged)} flagged entries…")
    inserted = update_all_pair_stats(course_id, semester_id, all_flagged)
    print(f"    ✔️ update_all_pair_stats inserted/updated {inserted} rows")

    # 4) Clear & rebuild student semester profiles
    StudentSemesterProfile.objects.filter(
        course_catalog_id=course_id,
        semester_id=semester_id,
    ).delete()
    print("  🔄 Cleared old StudentSemesterProfile rows")
    print("  🔄 Recomputing StudentSemesterProfile features…")
    t0 = time.time()
    bulk_recompute_semester_profiles(course_id, semester_id)
    print(f"    ✔️ Profiles recomputed in {time.time() - t0:.2f}s")

    # 5) Run student-level K-Means clustering
    print("  🔄 Running student-level K-Means…")
    t1 = time.time()
    student_km = run_kmeans_for_course_semester(course_id, semester_id)
    print(
        f"    ✔️ Student K-Means done in {time.time() - t1:.2f}s (k={student_km.n_clusters})"
    )

    # 6) Run the per-pair analytics pipeline:
    #    - populate_student_pair_stats
    #    - run_pair_level_clustering
    print("  🔄 Running data-analytics pipeline (pair stats + clustering)…")
    pairs_stats, population_stats = populate_student_pair_stats(course_id, semester_id)
    run_pair_level_clustering(course_id, semester_id)
    print(
        f"    ✔️ Data analytics done: {len(pairs_stats)} pairs, population={population_stats['total_pairs']}"
    )

    total = time.time() - start_ts
    print(f"✅ [PIPELINE] Finished full pipeline in {total:.2f}s\n")

    return JsonResponse(
        {
            "status": "success",
            "duration_s": total,
            "assignments_processed": len(assignments),
            "flagged_rows_upserted": inserted,
            "student_clusters": student_km.n_clusters,
            "pairs_stats_count": len(pairs_stats),
            "population_pairs": population_stats["total_pairs"],
        }
    )


@require_GET
def four_panel_pipeline_plot(request, course_id, semester_id):
    """
    2×2 figure.

      ┌───────────────────┬───────────────────┐
      │ 1) Histogram +    │ 2) Pie chart:     │
      │    Normal PDF     │    normal vs out. │
      ├───────────────────┼───────────────────┤
      │ 3) Q–Q plot for   │ 4) Tail CDF:      │
      │    checking       │    frac ≥ Z       │
      │    normality      │                   │
      └───────────────────┴───────────────────┘
    """
    # 1) Fetch data
    means = np.array(
        StudentPairSimilarityStatistics.objects.filter(
            course_catalog_id=course_id, semester_id=semester_id
        ).values_list("mean_similarity_score", flat=True)
    )
    zs = np.array(
        StudentPairSimilarityStatistics.objects.filter(
            course_catalog_id=course_id, semester_id=semester_id
        ).values_list("mean_z_score", flat=True)
    )
    if means.size == 0 or zs.size == 0:
        raise Http404("No similarity data to plot")

    n = zs.size
    mu, sigma = means.mean(), means.std()
    n_out = int((zs >= 2.0).sum())
    n_norm = n - n_out

    # Tail CDF
    sorted_z = np.sort(zs)
    cdf = np.arange(1, n + 1) / n

    # 2×2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # ── 1) Histogram + PDF ─────────────────────────────────────
    ax0 = axs[0, 0]
    ax0.hist(means, bins=30, density=True, alpha=0.6, edgecolor="k")
    xs = np.linspace(means.min(), means.max(), 200)
    ax0.plot(xs, stats.norm.pdf(xs, mu, sigma), "k--", label="Normal PDF")
    ax0.set_title(f"Mean Similarity Histogram (n={n}, μ={mu:.2f}, σ={sigma:.2f})")
    ax0.set_xlabel("Mean Similarity (%)")
    ax0.set_ylabel("Density")
    ax0.legend()

    # ── 2) Pie chart ────────────────────────────────────────────
    ax1 = axs[0, 1]
    wedges, texts, autotexts = ax1.pie(
        [n_norm, n_out],
        labels=["Normal (< 2.0)", "Outlier (≥ 2.0)"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["C0", "C3"],
        wedgeprops={"edgecolor": "k"},
    )
    ax1.set_title("Pairs by Z‑Score Region")

    # raw counts above
    fig.text(
        0.62,
        0.92,
        f"Normal: {n_norm}   Outlier: {n_out}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # outline % labels for contrast
    for txt in autotexts:
        txt.set_fontsize(14)
        txt.set_fontweight("bold")
        txt.set_color("white")
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])

    # ── 3) Q–Q plot ─────────────────────────────────────────────
    ax2 = axs[1, 0]
    stats.probplot(means, dist="norm", sparams=(mu, sigma), plot=ax2)
    ax2.get_lines()[1].set_color("red")  # the 45° reference line
    ax2.set_title("Normal Q–Q Plot of Mean Similarities")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")

    # ── 4) Tail CDF ─────────────────────────────────────────────
    ax3 = axs[1, 1]
    ax3.plot(sorted_z, 1 - cdf, marker=".", linestyle="none")
    ax3.axvline(2.0, color="r", linestyle="--")
    ax3.set_title("Fraction of Pairs ≥ Z")
    ax3.set_xlabel("Z‑Score")
    ax3.set_ylabel("Fraction ≥ Z")

    fig.tight_layout()

    # Render to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/png")


@require_GET
def kmeans_pairs_plot(request, course_id, semester_id):
    """
    1) Pull StudentPairSimilarityStatistics for course+sem.

    2) Build 9-feature matrix (incl. flagged_count & squares).
    3) Pick k via Gap-statistic, fit KMeans, PCA→2D.
    4) Color clusters with tab10 but force the MOST-SUSPICIOUS cluster red.
    5) List & highlight Top 20 student pairs in that cluster.
    6) Title includes course & semester names.
    """
    # fetch human-readable names
    catalog = CourseCatalog.objects.get(pk=course_id)
    sem = Semester.objects.get(pk=semester_id)

    qs = StudentPairSimilarityStatistics.objects.filter(
        course_catalog=catalog, semester=sem
    ).order_by("id")
    if not qs.exists():
        raise Http404("No pair stats available")

    # 1) Build N×9 feature matrix
    X = np.vstack(
        [
            [
                o.mean_similarity_score or 0.0,
                o.median_similarity_score or 0.0,
                o.similarity_std_dev or 0.0,
                o.max_z_score or 0.0,
                o.min_z_score or 0.0,
                o.total_z_score or 0.0,
                o.flagged_count or 0,
                (o.flagged_count or 0) ** 2,
                (o.max_z_score or 0.0) * abs(o.max_z_score or 0.0),
            ]
            for o in qs
        ]
    )

    # 2) Gap-statistic helpers
    def inertia(data, k):
        return KMeans(n_clusters=k, random_state=0).fit(data).inertia_

    def choose_k(data, k_max=6, B=5):
        n, d = data.shape
        mins, maxs = data.min(axis=0), data.max(axis=0)
        Wks = np.zeros(k_max)
        Wkbs = np.zeros((k_max, B))
        for k in range(1, k_max + 1):
            Wks[k - 1] = inertia(data, k)
            for b in range(B):
                ref = np.random.uniform(mins, maxs, size=(n, d))
                Wkbs[k - 1, b] = inertia(ref, k)
        logWks = np.log(Wks)
        logWkbs_mean = np.log(Wkbs).mean(axis=1)
        sk = np.sqrt(
            ((np.log(Wkbs) - logWkbs_mean[:, None]) ** 2).mean(axis=1)
        ) * np.sqrt(1 + 1 / B)
        gaps = logWkbs_mean - logWks
        for i in range(k_max - 1):
            if gaps[i] >= gaps[i + 1] - sk[i + 1]:
                return i + 1
        return k_max

    best_k = choose_k(X)

    # 3) Standardize & cluster
    mu, sigma = X.mean(0), X.std(0, ddof=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    km = KMeans(n_clusters=best_k, random_state=0).fit(Xs)
    labels = km.labels_

    # 4) PCA → 2D
    pca = PCA(n_components=2).fit(Xs)
    proj = pca.transform(Xs)

    # 5) Identify most-suspicious cluster by avg total_z_score
    unique = sorted(set(labels))
    avg_z = [
        np.mean([o.total_z_score for o, l in zip(qs, labels) if l == c]) for c in unique
    ]
    suspicious = unique[int(np.argmax(avg_z))]

    # build color array: tab10 for all, override suspicious → pure red
    cmap = plt.cm.get_cmap("tab10")
    norm = plt.Normalize(vmin=min(unique), vmax=max(unique))
    colors = cmap(norm(labels))
    colors[labels == suspicious] = np.array([1.0, 0.0, 0.0, 1.0])

    # draw scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=colors,
        edgecolors="k",
        linewidths=0.5,
        alpha=0.8,
        s=35,
    )
    # expand top margin
    fig.subplots_adjust(top=0.99)

    # 6) Title with course & semester
    ax.set_title(
        f"{catalog.name} — {sem.name}\n" f"Student-Pair Clusters (k={best_k})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")

    # legend (suspicious in red)
    handles = []
    for c in unique:
        face = "red" if c == suspicious else cmap(norm(c))
        handles.append(Patch(facecolor=face, edgecolor="k", label=f"Cluster {c}"))
    ax.legend(
        handles=handles,
        title="Cluster",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(unique),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    # 7) Pick Top 20 in that cluster by max_z_score
    candidates = [(i, o) for i, (o, l) in enumerate(zip(qs, labels)) if l == suspicious]
    top20 = sorted(candidates, key=lambda tup: tup[1].max_z_score, reverse=True)[:20]

    # 8) Side-panel: four-line header + list with red circles & black text
    tx, ty = 1.02, 1.02
    dy, dx = 0.045, 0.06

    # header
    ax.text(
        tx,
        ty,
        "Top 20 student pairs",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.text(
        tx,
        ty - dy,
        "in most‐suspicious cluster",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.text(
        tx,
        ty - 2 * dy,
        "ordered by max_z_score",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
        va="top",
        ha="left",
    )
    ax.text(
        tx,
        ty - 3 * dy,
        "(most‐suspicious = highest mean total_z_score)",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
        va="top",
        ha="left",
    )

    # list entries
    for rank, (_, o) in enumerate(top20, start=1):
        y = ty - 4 * dy - (rank - 1) * dy
        ax.text(
            tx,
            y,
            str(rank),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(boxstyle="circle,pad=0.2", fc="none", ec="red", lw=1),
            color="black",
            zorder=4,
        )
        ax.text(
            tx + dx,
            y,
            f"{o.student_a.first_name} {o.student_a.last_name} – "
            f"{o.student_b.first_name} {o.student_b.last_name} (z={o.max_z_score:.1f})",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            color="black",
        )

    # 9) Highlight & annotate on-plot
    idxs = [i for i, _ in top20]
    ax.scatter(
        proj[idxs, 0],
        proj[idxs, 1],
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        zorder=3,
    )
    for rank, (i, _) in enumerate(top20, start=1):
        x0, y0 = proj[i]
        ax.text(
            x0,
            y0,
            str(rank),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="red", lw=1),
            zorder=5,
        )

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/png")
