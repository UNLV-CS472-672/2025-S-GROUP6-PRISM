"""Cheating detection models for the application."""

from django.db import models
from assignments.models import Assignments, Submissions
from django.contrib.postgres.fields import ArrayField


class CheatingGroups(models.Model):
    """
    Represents a detected cheating group for a given assignment.

    Stores the cohesion score and analysis report path.
    """

    assignment = models.ForeignKey(
        Assignments,
        models.CASCADE,
    )
    cohesion_score = models.FloatField()
    analysis_report_path = models.TextField(unique=True)

    class Meta:
        """Model metadata configuration."""

        db_table_comment = "Recorded cheating groups for a given assignment."

    def __str__(self):
        """
        Return a readable representation of the cheating group.

        Includes assignment title and cohesion score.
        """
        return f"{self.assignment} - Cohesion Score: {self.cohesion_score:.2f}"


class CheatingGroupMembers(models.Model):
    """
    Represents a student's membership in a cheating group.

    Includes the distance metric used to determine clustering.
    """

    cheating_group = models.ForeignKey(
        "CheatingGroups",
        models.CASCADE,
    )
    student = models.ForeignKey(
        "courses.Students",
        models.CASCADE,
    )
    cluster_distance = models.FloatField()

    def __str__(self):
        """
        Return a readable representation of the group member.

        Displays the student and their cluster distance in the group.
        """
        return (
            f"{self.student} in Group {self.cheating_group.id} "
            f"(Distance: {self.cluster_distance:.2f})"
        )


class ConfirmedCheaters(models.Model):
    """
    Represents a confirmed cheating instance for a student.

    Links a student to an assignment with a confirmed date and
    threshold value used for confirmation.
    """

    confirmed_date = models.DateField()
    threshold_used = models.IntegerField()
    assignment = models.ForeignKey(
        Assignments,
        models.CASCADE,
    )
    student = models.ForeignKey(
        "courses.Students",
        models.CASCADE,
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (("student", "assignment"),)

    def __str__(self):
        """
        Return a readable representation of the confirmed cheating case.

        Includes student name, assignment title, and confirmation date.
        """
        return (
            f"{self.student} - {self.assignment} " f"(Confirmed: {self.confirmed_date})"
        )


class FlaggedStudents(models.Model):
    """
    Represents a student flagged for potential misconduct.

    Links a flagged student to a professor and a similarity record,
    with a flag indicating generative AI usage.
    """

    professor = models.ForeignKey(
        "courses.Professors",
        models.CASCADE,
    )
    student = models.ForeignKey(
        "courses.Students",
        models.CASCADE,
    )
    similarity = models.ForeignKey(
        "SubmissionSimilarityPairs",
        models.CASCADE,
    )
    generative_ai = models.BooleanField()

    class Meta:
        """Model metadata configuration."""

        unique_together = (("student", "similarity"),)

    def __str__(self):
        """
        Return a readable representation of the flagged student.

        Includes student, similarity pair, and whether AI was involved.
        """
        ai_flag = "AI" if self.generative_ai else "Manual"
        return f"{self.student} flagged by {self.professor} ({ai_flag})"


class SubmissionSimilarityPairs(models.Model):
    """
    Represents a detected similarity between two student submissions.

    Includes the assignment, matched file name, similarity percentage,
    and a unique match ID.
    """

    assignment = models.ForeignKey(
        Assignments,
        models.CASCADE,
    )
    file_name = models.CharField(max_length=50)
    submission_id_1 = models.ForeignKey(
        Submissions,
        models.CASCADE,
        db_column="submission_id_1",
    )
    submission_id_2 = models.ForeignKey(
        Submissions,
        models.CASCADE,
        db_column="submission_id_2",
        related_name=("submissionsimilaritypairs_submission_id_2_set"),
    )
    match_id = models.BigIntegerField()
    percentage = models.IntegerField()

    class Meta:
        """Model metadata configuration."""

        unique_together = (("submission_id_1", "submission_id_2", "assignment"),)

    def __str__(self):
        """
        Return a readable representation of the similarity pair.

        Displays the assignment, the two submissions and their
        similarity percentage.
        """
        return (
            f"{self.assignment}: "
            f"{self.submission_id_1} ↔ {self.submission_id_2} "
            f"({self.percentage}%)"
        )


class LongitudinalCheatingGroups(models.Model):
    """
    Represents a group of students flagged for repeated cheating patterns.

    This model captures a longitudinal score across multiple assignments.
    """

    score = models.FloatField()

    def __str__(self):
        """
        Return a readable representation of the cheating group.

        Displays the group ID and its score.
        """
        return f"Longitudinal Group {self.id} " f"(Score: {self.score:.2f})"


class LongitudinalCheatingGroupMembers(models.Model):
    """
    Represents a student within a longitudinal cheating group.

    Includes whether the student is a core member and how often
    they appeared in flagged groups.
    """

    longitudinal_cheating_group = models.ForeignKey(
        "LongitudinalCheatingGroups",
        models.CASCADE,
    )
    student = models.ForeignKey(
        "courses.Students",
        models.CASCADE,
    )
    is_core_member = models.BooleanField()
    appearance_count = models.IntegerField()

    def __str__(self):
        """
        Return a readable representation of a group member.

        Includes the student, group ID, and whether they are core.
        """
        role = "Core" if self.is_core_member else "Peripheral"
        return (
            f"{self.student} in Group "
            f"{self.id} ({role}, "
            f"{self.appearance_count} appearances)"
        )


class LongitudinalCheatingGroupInstances(models.Model):
    """
    Links a single cheating group to a longitudinal cheating group.

    Used to associate short-term cheating events with longer-term
    behavioral patterns.
    """

    cheating_group = models.ForeignKey(
        CheatingGroups,
        models.CASCADE,
    )
    longitudinal_cheating_group = models.ForeignKey(
        "LongitudinalCheatingGroups",
        models.CASCADE,
    )

    def __str__(self):
        """
        Return a readable representation of the group linkage.

        Displays the short-term and longitudinal group IDs.
        """
        return f"CheatingGroup {self.id} → " f"LongitudinalGroup {self.id}"


class AssignmentReport(models.Model):
    """Stores summary statistics for one Assignment’s similarity report."""

    assignment = models.ForeignKey(
        "assignments.Assignments",
        on_delete=models.CASCADE,
        related_name="reports",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    mu = models.FloatField(help_text="Population mean of all similarity scores")
    sigma = models.FloatField(help_text="Population standard deviation")
    variance = models.FloatField(help_text="Population variance (sigma^2)")

    class Meta:
        """Model metadata configuration."""

        ordering = ["-created_at"]

    def __str__(self):
        """Return a human‐readable summary of the report."""
        return (
            f"Report for Assignment {self.assignment_id} "
            f"@ {self.created_at:%Y-%m-%d %H:%M} — "
            f"μ={self.mu:.2f}, σ={self.sigma:.2f}"
        )


class StudentReport(models.Model):
    """Stores per-student inference results for one AssignmentReport."""

    report = models.ForeignKey(
        AssignmentReport,
        on_delete=models.CASCADE,
        related_name="student_reports",
    )
    submission = models.ForeignKey(
        "assignments.Submissions",
        on_delete=models.CASCADE,
        help_text="The student submission referenced",
    )
    mean_similarity = models.FloatField(help_text="Student’s sample mean similarity")
    z_score = models.FloatField(help_text="The student’s z-score of mean similarity")
    ci_lower = models.FloatField(help_text="Lower bound of 95% CI for mean similarity")
    ci_upper = models.FloatField(help_text="Upper bound of 95% CI for mean similarity")

    class Meta:
        """Model metadata configuration."""

        unique_together = ("report", "submission")
        ordering = ["-z_score"]
        indexes = [
            models.Index(fields=["submission"]),
            # index z_score and mean_similarity so filters are fast
            models.Index(fields=["z_score"]),
            models.Index(fields=["mean_similarity"]),
        ]

    def __str__(self):
        """Return a human‐readable summary of this student’s inference."""
        return (
            f"StudentReport(sub={self.submission_id}, "
            f"z={self.z_score:.2f}, "
            f"CI=[{self.ci_lower:.1f},{self.ci_upper:.1f}])"
        )


class StudentSemesterProfile(models.Model):
    """Pre‑computed semester‑level metrics for each student."""

    student = models.ForeignKey(
        "courses.Students",
        on_delete=models.CASCADE,
        help_text="Which student these features belong to",
    )
    course_catalog = models.ForeignKey(
        "courses.CourseCatalog",
        on_delete=models.CASCADE,
        help_text="The course this profile belongs to",
    )
    semester = models.ForeignKey(
        "courses.Semester",
        on_delete=models.CASCADE,
        help_text="The semester this profile covers",
    )

    # ─────────────── raw summary stats ───────────────
    avg_z_score = models.FloatField(
        help_text="Average of that student's z‑scores across assignments"
    )
    max_z_score = models.FloatField(help_text="Maximum single‑assignment z‑score")
    num_flagged_assignments = models.PositiveIntegerField(
        help_text="How many assignments where z > threshold"
    )

    mean_similarity_variance = models.FloatField(
        help_text="Population variance of per‑assignment mean similarities"
    )
    mean_similarity_skewness = models.FloatField(
        help_text="Skewness of per‑assignment mean similarities"
    )
    mean_similarity_kurtosis = models.FloatField(
        help_text="Kurtosis of per‑assignment mean similarities"
    )

    high_similarity_fraction = models.FloatField(
        help_text=(
            "Fraction of *all* pairwise comparisons > threshold " "across the semester"
        )
    )

    # ───────── full 7‑dim feature vector ─────────
    feature_vector = ArrayField(
        base_field=models.FloatField(),
        size=7,
        help_text="[avg_z, max_z, num_flagged, sim_var, sim_skew, sim_kurt, high_frac]",
    )

    # ───────── clustering output ─────────
    cluster_label = models.IntegerField(
        null=True,
        blank=True,
        help_text="Cluster assignment from the latest KMeans run",
    )

    last_updated = models.DateTimeField(
        auto_now=True,
        help_text="When these features were last recomputed",
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (("student", "course_catalog", "semester"),)
        ordering = ["-last_updated"]

    def __str__(self):
        """Return a human‑readable summary of this student’s profile."""
        return (
            f"{self.student} | {self.course_catalog} @ {self.semester}: "
            f"avg_z={self.avg_z_score:.2f}, max_z={self.max_z_score:.2f}, "
            f"cluster={self.cluster_label}"
        )


class PairFlagStat(models.Model):
    """
    Represents a pair of students and their similarity statistics.

    This model is used to track the similarity scores and
    z-scores between two students over time.
    It is used to identify potential cheating behavior
    and to analyze the relationships between students.
    It includes fields for the course catalog, semester,
    students involved, and various statistics related to
    their submissions.
    It also includes methods to calculate the proportion
    of flagged assignments and the mean similarity
    and z-scores for the pair.
    """

    course_catalog = models.ForeignKey(
        "courses.CourseCatalog", on_delete=models.CASCADE
    )
    semester = models.ForeignKey("courses.Semester", on_delete=models.CASCADE)
    student_a = models.ForeignKey(
        "courses.Students", on_delete=models.CASCADE, related_name="+"
    )
    student_b = models.ForeignKey(
        "courses.Students", on_delete=models.CASCADE, related_name="+"
    )

    # how many assignments both turned in
    assignments_shared = models.PositiveIntegerField(default=0)
    # how many times they landed in the high‑z / red zone
    flagged_count = models.PositiveIntegerField(default=0)
    # cumulative sum of their raw similarity %s
    total_similarity = models.FloatField(default=0.0)
    # cumulative sum of their z‑scores
    total_z_score = models.FloatField(default=0.0)
    max_z_score = models.FloatField(default=0.0)

    kmeans_label = models.SmallIntegerField(
        null=True,
        blank=True,
        help_text="Cluster ID from latest KMeans run",
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (
            "course_catalog",
            "semester",
            "student_a",
            "student_b",
        )

    @property
    def proportion(self) -> float:
        """Calculate the proportion of flagged assignments."""
        return (
            (self.flagged_count / self.assignments_shared)
            if self.assignments_shared
            else 0.0
        )

    @property
    def mean_similarity(self) -> float:
        """Calculate the mean similarity score."""
        return (
            (self.total_similarity / self.assignments_shared)
            if self.assignments_shared
            else 0.0
        )

    @property
    def mean_z_score(self) -> float:
        """Calculate the mean z-score."""
        return (
            (self.total_z_score / self.assignments_shared)
            if self.assignments_shared
            else 0.0
        )


class FlaggedStudentPair(models.Model):
    """
    Represents a pair of students flagged for potential misconduct.

    This model captures the similarity scores and z-scores
    between two students over time.
    It is used to identify potential cheating behavior
    and to analyze the relationships between students.
    """

    course_catalog = models.ForeignKey(
        "courses.CourseCatalog", on_delete=models.CASCADE
    )
    semester = models.ForeignKey("courses.Semester", on_delete=models.CASCADE)
    student_a = models.ForeignKey(
        "courses.Students", related_name="+", on_delete=models.CASCADE
    )
    student_b = models.ForeignKey(
        "courses.Students", related_name="+", on_delete=models.CASCADE
    )
    mean_similarity = models.FloatField()
    max_similarity = models.FloatField()
    mean_z_score = models.FloatField()
    max_z_score = models.FloatField()
    flagged_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Model metadata configuration."""

        unique_together = ("course_catalog", "semester", "student_a", "student_b")


class StudentPairSimilarityStatistics(models.Model):
    """Represents the similarity statistics between two students."""

    course_catalog = models.ForeignKey(
        "courses.CourseCatalog", on_delete=models.CASCADE
    )
    semester = models.ForeignKey("courses.Semester", on_delete=models.CASCADE)
    student_a = models.ForeignKey(
        "courses.Students", related_name="+", on_delete=models.CASCADE
    )
    student_b = models.ForeignKey(
        "courses.Students", related_name="+", on_delete=models.CASCADE
    )

    assignments_shared = models.PositiveIntegerField()

    # need per assignment z-scores to calculate
    flagged_count = models.PositiveIntegerField(default=0, blank=True, null=True)

    mean_similarity_score = models.FloatField()
    median_similarity_score = models.FloatField()
    similarity_std_dev = models.FloatField()
    similarity_variance = models.FloatField()
    min_similarity_score = models.FloatField()
    max_similarity_score = models.FloatField()

    mean_z_score = models.FloatField()

    # need per assignment z-scores to calculate
    median_z_score = models.FloatField(blank=True, null=True)

    min_z_score = models.FloatField()
    max_z_score = models.FloatField()

    total_similarity = models.FloatField()
    total_z_score = models.FloatField()

    cluster_id = models.PositiveSmallIntegerField(null=True, blank=True)
    pair_flagged = models.BooleanField(default=False)
    ta_notes = models.TextField(blank=True, null=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Model metadata configuration."""

        unique_together = ("course_catalog", "semester", "student_a", "student_b")


class PopulationSimilarityStatistics(models.Model):
    """
    Represents the similarity statistics for a population of students.

    This model captures the overall similarity statistics
    for all students in a course and semester.
    It is used to analyze the relationships between students
    and to identify potential cheating behavior.
    """

    course_catalog = models.ForeignKey(
        "courses.CourseCatalog", on_delete=models.CASCADE
    )
    semester = models.ForeignKey("courses.Semester", on_delete=models.CASCADE)

    total_unique_pairs = models.PositiveIntegerField(default=0)
    total_flagged_count = models.PositiveIntegerField(default=0, blank=True, null=True)

    mean_similarity_score = models.FloatField()
    median_similarity_score = models.FloatField()
    similarity_std_dev = models.FloatField()
    similarity_variance = models.FloatField()
    min_similarity_score = models.FloatField(blank=True, null=True)
    max_similarity_score = models.FloatField(blank=True, null=True)

    mean_z_score = models.FloatField()
    median_z_score = models.FloatField()
    min_z_score = models.FloatField(blank=True, null=True)
    max_z_score = models.FloatField(blank=True, null=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Model metadata configuration."""

        unique_together = ("course_catalog", "semester")
