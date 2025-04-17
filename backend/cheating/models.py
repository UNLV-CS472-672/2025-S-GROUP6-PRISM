"""Cheating detection models for the application."""

from django.db import models
from assignments.models import Assignments, Submissions


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
