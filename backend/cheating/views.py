"""Views for the Cheating app with enhanced filtering, ordering, search, and pagination."""

from django.db import connection
import io
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from courses.models import CourseCatalog, Semester
import matplotlib
from matplotlib import pyplot as plt
import time
import numpy as np
from django.db import transaction
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework import filters, viewsets
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
    get_all_similarity_pairs,
    turn_pairs_to_dict,
    compute_pairs_statistics,
)
from .services import generate_report as generate_report_service
from django.views.decorators.http import require_GET
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


@require_GET
def kmeans_pairs_plot(request, course_id, semester_id):
    """
    Generate a 2D PCA scatter of submission-pair clusters using an 8-dim feature.

    Feature vector. Steps:
      1) Fetch all PairFlagStat rows (with students) in one query.
      2) Build an 8-dim matrix per pair:
         [ mean_z*|mean_z|,
           max_z*|max_z|,
           mean_similarity,
           proportion*100,
           student_label_sum,
           flagged_count²,
           total_similarity,
           total_z_score ]
      3) Standardize, then reduce to 2D via PCA.
      4) Identify the “red” cluster by PC1 centroid. If too many points,
         highlight only the single worst pair; otherwise show Low/Med/High.
      5) Draw convex hulls or circles, annotate as needed.
      6) Return the plot as a PNG HTTP response.
    """
    # 1) Load every pair stat with related students in one go
    pairs = list(
        PairFlagStat.objects.filter(
            course_catalog_id=course_id,
            semester_id=semester_id,
        ).select_related("student_a", "student_b")
    )
    if not pairs:
        raise Http404("No pair statistics found.")

    # 1a) Build a student_id → cluster_label lookup in one query
    profiles = StudentSemesterProfile.objects.filter(
        course_catalog_id=course_id,
        semester_id=semester_id,
    ).only("student_id", "cluster_label")
    label_map = {prof.student_id: (prof.cluster_label or 0) for prof in profiles}

    # 2) Assemble the raw 8-dim feature matrix
    X_raw = np.zeros((len(pairs), 8), dtype=float)
    for idx, ps in enumerate(pairs):
        # amplify z-scores by squaring with sign
        X_raw[idx, 0] = ps.mean_z_score * abs(ps.mean_z_score)
        X_raw[idx, 1] = ps.max_z_score * abs(ps.max_z_score)
        # raw mean similarity and proportion of flagged comparisons
        X_raw[idx, 2] = ps.mean_similarity
        X_raw[idx, 3] = ps.proportion * 100.0
        # link pair to combined student risk labels
        X_raw[idx, 4] = label_map.get(ps.student_a_id, 0) + label_map.get(
            ps.student_b_id, 0
        )
        # cumulative stats from PairFlagStat
        X_raw[idx, 5] = ps.flagged_count**2
        X_raw[idx, 6] = ps.total_similarity
        X_raw[idx, 7] = ps.total_z_score

    # 3) Standardize features to mean=0, std=1 for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3a) Reduce to 2D with PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    dim1, dim2 = coords[:, 0], coords[:, 1]
    var1, var2 = pca.explained_variance_ratio_[:2] * 100

    # 4) Identify the “red” cluster by highest PC1 centroid
    labels = np.array([ps.kmeans_label for ps in pairs])
    unique_labels = sorted(set(labels))
    centroids = {cluster: dim1[labels == cluster].mean() for cluster in unique_labels}
    red_label = max(centroids, key=centroids.get)
    red_mask = labels == red_label
    red_count = int(red_mask.sum())

    # 5) Decide whether to highlight one worst pair or show clusters
    red_threshold = 30
    if red_count > red_threshold:
        # too many reds → mark only the single worst composite score
        composite_scores = X_raw[:, 0] + X_raw[:, 1] + X_raw[:, 3]
        worst_idx = int(np.argmax(composite_scores))
        highlight_mask = np.zeros_like(labels, dtype=bool)
        highlight_mask[worst_idx] = True

        low_mask = ~highlight_mask
        med_mask = np.zeros_like(labels, dtype=bool)
        high_mask = highlight_mask

        low_count, med_count, high_count = (int(low_mask.sum()), 0, 1)
        highlight_color = "gold"
        highlight_label = "Outlier Candidate (n=1)"
        show_names = False
    else:
        # normal Low/Med/High split
        low_label = min(centroids, key=centroids.get)
        med_labels = [lbl for lbl in unique_labels if lbl not in (low_label, red_label)]

        low_mask = labels == low_label
        med_mask = np.isin(labels, med_labels)
        high_mask = red_mask

        low_count = int(low_mask.sum())
        med_count = int(med_mask.sum())
        high_count = red_count

        highlight_color = "darkred"
        highlight_label = f"High Intensity (n={high_count})"
        show_names = True

    # 6) Prepare the figure and axes
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.00, 0.04, 0.78, 0.96])

    def plot_group(mask, color, marker, legend_text):
        """Plot one cluster: scatter + hull or circle."""
        xs, ys = dim1[mask], dim2[mask]
        ax.scatter(
            xs,
            ys,
            c=color,
            marker=marker,
            s=100,
            edgecolor="k",
            alpha=0.8,
            label=legend_text,
        )
        points = list(zip(xs, ys))
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hx, hy = zip(*(points[i] for i in hull.vertices))
                ax.fill(hx, hy, color=color, alpha=0.2)
            except QhullError:
                pass
        elif points:
            cx, cy = np.mean(xs), np.mean(ys)
            radius = (np.hypot(xs - cx, ys - cy).max() if len(xs) > 1 else 0.5) * 1.5
            ax.add_patch(Circle((cx, cy), radius=radius, color=color, alpha=0.2))

    # 7) Draw Low, Medium (if any), and High/outlier clusters
    plot_group(low_mask, "green", "o", f"Low Intensity (n={low_count})")
    if med_mask.any():
        plot_group(med_mask, "gold", "s", f"Medium Intensity (n={med_count})")
    plot_group(high_mask, highlight_color, "^", highlight_label)

    # 8) Annotate high cluster members if required
    if show_names and high_count:
        high_indices = np.where(high_mask)[0]
        for idx_num, idx in enumerate(high_indices, start=1):
            ax.annotate(
                str(idx_num),
                xy=(dim1[idx], dim2[idx]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color=highlight_color,
            )
        sidebar_names = [
            f"{pairs[idx].student_a.first_name} "
            f"{pairs[idx].student_a.last_name} & "
            f"{pairs[idx].student_b.first_name} "
            f"{pairs[idx].student_b.last_name}"
            for idx in high_indices
        ]
        y_start, y_end = 0.98, 0.04
        step = (y_start - y_end) / (len(sidebar_names) + 1)
        for line_num, text in enumerate(sidebar_names, start=1):
            fig.text(
                0.79,
                y_start - line_num * step,
                f"{line_num}: {text} "
                f"(max_z={pairs[high_indices[line_num - 1]].max_z_score:.2f})",
                transform=fig.transFigure,
                va="top",
                ha="left",
                fontsize=11,
                fontweight="bold",
                color=highlight_color,
                family="monospace",
            )
    elif not show_names:
        fig.text(
            0.79,
            0.5,
            "Not enough evidence of anomalies",
            transform=fig.transFigure,
            va="center",
            ha="left",
            fontsize=14,
            fontstyle="italic",
        )

    # 9) Final styling: axes labels, title, grid, legend
    ax.set_xlabel(f"Dimension 1 ({var1:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"Dimension 2 ({var2:.1f}% var)", fontsize=12)
    course = get_object_or_404(CourseCatalog, pk=course_id)
    sem = get_object_or_404(Semester, pk=semester_id)
    ax.set_title(f"K-Means Clusters → {course} Pairs → {sem}", fontsize=16, pad=15)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    fig.subplots_adjust(bottom=0.06)
    legend_cols = 2 if not med_mask.any() else 3
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.18),
        ncol=legend_cols,
        title="Pair Groups",
        frameon=True,
        fontsize=11,
        title_fontsize=13,
    )

    # 10) Render to PNG and return HTTP response
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return HttpResponse(buffer.read(), content_type="image/png")


def _generate_one(assignment_id):
    """
    Help for ThreadPoolExecutor: close the old DB connection.

    then run the report service in a fresh one.
    """
    connection.close()
    return generate_report_service(assignment_id)


@require_GET
def run_full_pipeline(request, course_id, semester_id):
    """
    0) Clear old AssignmentReport rows for this course+semester.

    1) Clear old PairFlagStat rows.
    2) Parallel-generate each assignment report and collect flagged pairs.
    3) Bulk-upsert PairFlagStat entries.
    4) Clear and recompute StudentSemesterProfile.
    5) Run student-level K-Means and capture the model.
    6) Run pair-level K-Means using student labels.
    """
    start_ts = time.time()
    print(
        f"\n🚀 [PIPELINE] Starting full pipeline "
        f"for course={course_id}, semester={semester_id}"
    )

    # 0) Remove every previous AssignmentReport (and its StudentReports)
    #    so we always start fresh and never mix old data with new.
    AssignmentReport.objects.filter(
        assignment__course_catalog_id=course_id,
        assignment__semester_id=semester_id,
    ).delete()
    print("  🔄 Cleared old AssignmentReport rows")

    # 1) Remove every previous PairFlagStat row for this course/semester,
    #    so our later bulk-upsert won’t leave stale entries behind.
    PairFlagStat.objects.filter(
        course_catalog_id=course_id,
        semester_id=semester_id,
    ).delete()
    print("  🔄 Cleared old PairFlagStat rows")

    # 2) Fetch all assignments once in memory. If none exist, bail out.
    assignments = list(
        Assignments.objects.filter(
            course_catalog_id=course_id,
            semester_id=semester_id,
        )
    )
    if not assignments:
        raise Http404("No assignments for that course+semester.")
    print(
        f"  📋 Found {len(assignments)} assignments; " "generating reports in parallel…"
    )

    # 2a) Generate each report concurrently; collect all flagged-pair data.
    all_flagged = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Schedule _generate_one for each assignment ID.
        futures = {pool.submit(_generate_one, a.id): a for a in assignments}
        for fut in as_completed(futures):
            a = futures[fut]
            try:
                report, flagged = fut.result()
            except Exception as exc:
                # If one assignment fails, log and continue with others.
                print(f"  ❌ generate_report({a.id}) failed:", exc)
                continue

            if report:
                print(f"    ▶ report.id={report.id}, " f"flagged_pairs={len(flagged)}")
                all_flagged.extend(flagged)

    # 3) Perform a single bulk-upsert of all pair stats.
    #    This replaces the old N+1 raw insert loops.
    print(f"  🔄 Bulk-upserting {len(all_flagged)} flagged entries…")
    inserted = update_all_pair_stats(course_id, semester_id, all_flagged)
    print(f"    ✔️ update_all_pair_stats inserted/updated {inserted} rows")

    # 4) Clear and rebuild student semester profiles in one bulk pass.
    StudentSemesterProfile.objects.filter(
        course_catalog_id=course_id,
        semester_id=semester_id,
    ).delete()
    print("  🔄 Cleared old StudentSemesterProfile rows")
    print("  🔄 Recomputing StudentSemesterProfile features…")
    t0 = time.time()
    bulk_recompute_semester_profiles(course_id, semester_id)
    print(f"  ✔️ Profiles recomputed in {time.time() - t0:.2f}s")

    # 5) Run the student-level K-Means clustering.
    print("  🔄 Running student K-Means…")
    t1 = time.time()
    student_km = run_kmeans_for_course_semester(course_id, semester_id)
    print(
        "  ✔️ Student K-Means done in "
        f"{time.time() - t1:.2f}s (k={student_km.n_clusters})"
    )

    # 6) Run the pair-level K-Means clustering using the student model.
    print("  🔄 Running pair K-Means…")
    t2 = time.time()
    run_kmeans_for_pair_stats(
        course_id=course_id,
        semester_id=semester_id,
        student_km=student_km,
        n_clusters=6,
        random_state=0,
        b_refs=10,
    )
    print(f"  ✔️ Pair K-Means done in {time.time() - t2:.2f}s")

    # Final: calculate and return the total duration of the pipeline.
    total = time.time() - start_ts
    print(f"✅ [PIPELINE] Finished full pipeline in {total:.2f}s\n")
    return JsonResponse({"status": "success", "duration_s": total})



















logger = logging.getLogger(__name__)

@require_GET
def data_analytics_pipeline(
    request: HttpRequest,
    course_id: int,
    semester_id: int,
) -> JsonResponse:
    """
    Endpoint to trigger the full data analytics pipeline for a course and semester.

    Steps:
      1. Fetch all similarity pairs from the DB.
      2. Turn them into an in-memory dict of student-pair → [(score, assignment), …].
      3. Compute per-pair and population statistics (including z-scores).
      4. Return everything as JSON, including timing.

    Returns:
        JsonResponse containing:
          - pairs: Dict[Tuple[int,int], Dict[str, Any]]
          - population: Dict[str, Any]
          - duration_secs: float
    """
    start_ts = time.time()
    logger.info(f"[PIPELINE] Starting for course={course_id}, semester={semester_id}")

    # 1) Fetch raw pairs
    raw_pairs = get_all_similarity_pairs(course_id, semester_id)
    if not raw_pairs:
        return JsonResponse(
            {'error': 'No student pairs found for that course+semester.'},
            status=404
        )

    # 2) Build in-memory dict
    pairs_dict = turn_pairs_to_dict(raw_pairs)

    # 3) Compute stats
    per_pair_stats, population_stats = compute_pairs_statistics(pairs_dict)
    if not per_pair_stats:
        return JsonResponse(
            {'error': 'No pair statistics computed for that course+semester.'},
            status=404
        )

    duration = time.time() - start_ts
    logger.info(
        "[PIPELINE] Completed in %.2f sec — %d pairs; μ=%.2f; σ=%.2f",
        duration,
        population_stats['total_pairs'],
        population_stats['population_mean'],
        population_stats['population_standard_deviation'],
    )

    # necessary to convert the per_pair_stats dict into a list of dicts
    # else JsonResponse will not serialize it properly
    pairs_list = [
        {
        'student_id_1': a,
        'student_id_2': b,
        **stats
        } for (a, b), stats in per_pair_stats.items()
    ]

    # 4) Respond
    return JsonResponse({
        'pairs': pairs_list,
        'population': population_stats,
        'duration_secs': round(duration, 3),
    })