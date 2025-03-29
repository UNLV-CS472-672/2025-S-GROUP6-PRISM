"""Assignment-related models for students, professors, and submissions.

This module defines models to manage academic workflows including assignments,
submissions, flagged results, and confirmed cheating detections.
"""

from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator


class Student(models.Model):
    """Represent a student entity with personal and academic information.

    Attributes:
        email (EmailField): Unique email address of the student.
        codeGrade_id (IntegerField): Unique CodeGrade identifier.
        username (CharField): Optional CodeGrade username.
        first_name (CharField): First name of the student.
        last_name (CharField): Last name of the student.

    Methods:
        __str__(): Return string as "FirstName LastName (Email)".
    """

    email = models.EmailField(max_length=100, unique=True)
    codeGrade_id = models.IntegerField(unique=True)
    username = models.CharField(max_length=50, blank=True, null=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)

    def __str__(self):
        """Return string representation of the student."""
        return f"{self.first_name} {self.last_name} ({self.email})"


class Assignment(models.Model):
    """Represent an assignment for a specific class and professor.

    Attributes:
        class_instance (ForeignKey): Link to class.
        professor (ForeignKey): Link to professor.
        assignment_number (IntegerField): Assignment sequence number.
        title (CharField): Title of the assignment.
        due_date (DateField): Due date of the assignment.

    Methods:
        __str__(): Return formatted title string.
    """

    class_instance = models.ForeignKey("courses.Class", on_delete=models.CASCADE)
    professor = models.ForeignKey("courses.Professor", on_delete=models.CASCADE)
    assignment_number = models.IntegerField()
    title = models.CharField(max_length=100)
    due_date = models.DateField()

    class Meta:
        """Meta options for Assignment."""

        unique_together = ("class_instance", "assignment_number")

    def __str__(self):
        """Return string representation of the assignment."""
        return f"Assignment {self.assignment_number}: {self.title}"


class Submission(models.Model):
    """Represent a student's submission for an assignment.

    Attributes:
        student (ForeignKey): Link to student.
        assignment (ForeignKey): Link to assignment.
        grade (IntegerField): Grade from 0–100.
        created_at (DateField): Auto-set date of submission.
        flagged (BooleanField): Whether the submission is flagged.
        professor (ForeignKey): Link to professor.

    Methods:
        __str__(): Return formatted summary string.
    """

    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE)
    grade = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    created_at = models.DateField(auto_now_add=True)
    flagged = models.BooleanField(default=False)
    professor = models.ForeignKey("courses.Professor", on_delete=models.CASCADE)

    def __str__(self):
        """Return string representation of the submission."""
        return f"Submission by {self.student} for {self.assignment}"


class FlaggedSubmission(models.Model):
    """Represent a flagged submission with high similarity.

    Attributes:
        submission (ForeignKey): Link to related submission.
        file_name (CharField): Name of submitted file.
        percentage (IntegerField): Similarity percentage.
        similarity_with (ManyToManyField): Related students.

    Methods:
        __str__(): Return formatted flagged summary.
    """

    submission = models.ForeignKey(Submission, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=50)
    percentage = models.IntegerField(
        validators=[MinValueValidator(20), MaxValueValidator(100)]
    )
    similarity_with = models.ManyToManyField(
        Student, related_name="similar_submissions"
    )

    def __str__(self):
        """Return string representation of the flagged submission."""
        return f"Flagged Submission: {self.file_name} ({self.percentage}%)"


class FlaggedStudent(models.Model):
    """Represent a student flagged multiple times for exceeding threshold.

    Attributes:
        student (ForeignKey): Link to student.
        professor (ForeignKey): Link to professor.
        times_over_threshold (IntegerField): Count of times flagged.

    Methods:
        __str__(): Return summary string.
    """

    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    professor = models.ForeignKey("courses.Professor", on_delete=models.CASCADE)
    times_over_threshold = models.IntegerField(default=0)

    def __str__(self):
        """Return string representation of flagged student."""
        return (
            f"Flagged Student: {self.student} "
            f"flagged {self.times_over_threshold} times"
        )


class ConfirmedCheater(models.Model):
    """Represent a student confirmed as cheating.

    Attributes:
        student (ForeignKey): Link to student.
        professor (ForeignKey): Link to professor.
        confirmed_date (DateField): Date cheating was confirmed.
        threshold_used (IntegerField): Similarity threshold used.

    Methods:
        __str__(): Return formatted confirmation summary.
    """

    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    professor = models.ForeignKey("courses.Professor", on_delete=models.CASCADE)
    confirmed_date = models.DateField(auto_now_add=True)
    threshold_used = models.IntegerField(default=40)

    def __str__(self):
        """Return string representation of the confirmed cheater."""
        return (
            f"Confirmed Cheater: {self.student} "
            f"on {self.confirmed_date} "
            f"(Threshold: {self.threshold_used}%)"
        )
