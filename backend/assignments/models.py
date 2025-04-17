"""Assignment-related models for students, professors, and submissions.

This module defines models to manage academic workflows including assignments,
submissions, flagged results, and confirmed cheating detections.
"""

from django.db import models


class Assignments(models.Model):
    """
    Represents an assignment for a given course catalog and semester.

    Includes assignment number, title, due date, language, directory paths,
    and flags for base code and policy.
    """

    course_catalog = models.ForeignKey(
        "courses.CourseCatalog",
        models.CASCADE,
    )
    semester = models.ForeignKey(
        "courses.Semester",
        models.CASCADE,
    )
    assignment_number = models.SmallIntegerField()
    title = models.TextField()
    due_date = models.DateField()
    pdf_filepath = models.TextField(
        unique=True,
        blank=True,
        null=True,
    )
    has_base_code = models.BooleanField()
    moss_report_directory_path = models.TextField(unique=True)
    bulk_ai_directory_path = models.TextField(unique=True)
    language = models.TextField()
    has_policy = models.BooleanField()

    class Meta:
        """Model metadata configuration."""

        unique_together = (("course_catalog", "semester", "assignment_number"),)

    def __str__(self):
        """
        Return a readable representation of the assignment.

        Includes course catalog, semester, assignment number, and title.
        """
        return (
            f"{self.course_catalog} [{self.semester}] – "
            f"Assignment {self.assignment_number}: {self.title}"
        )


class Submissions(models.Model):
    """
    Stores metadata for individual student submissions.

    Includes grade, file path, assignment link, and whether the submission
    was flagged.
    """

    grade = models.DecimalField(max_digits=5, decimal_places=2)
    created_at = models.DateField(blank=True, null=True)
    flagged = models.BooleanField()
    assignment = models.ForeignKey(
        "Assignments",
        models.CASCADE,
    )
    student = models.ForeignKey(
        "courses.Students",
        models.CASCADE,
    )
    course_instance = models.ForeignKey(
        "courses.CourseInstances",
        models.CASCADE,
    )
    file_path = models.TextField(
        unique=True,
        db_comment=("Relative path from the bulk submission directory."),
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (("assignment", "student"),)
        db_table_comment = "Stores relevant data for individual student submissions."

    def __str__(self):
        """
        Return a readable representation of the submission.

        Includes student, assignment, and grade.
        """
        return f"{self.student} - {self.assignment} (Grade: {self.grade})"


class BaseFiles(models.Model):
    """
    Represents base files provided to students for assignments.

    These files are used to inform the MOSS similarity report generator to
    exclude common starter files when comparing submissions.
    """

    assignment = models.ForeignKey(
        "Assignments",
        models.CASCADE,
    )
    file_name = models.TextField()
    file_path = models.TextField(unique=True)

    class Meta:
        """Model metadata configuration."""

        unique_together = (("assignment", "file_name"),)
        db_table_comment = (
            "Lists base files given to students and used to exclude shared "
            "files in MOSS similarity reports."
        )

    def __str__(self):
        """
        Return a readable representation of the base file.

        Includes the assignment and file name.
        """
        return f"{self.assignment} - {self.file_name}"


class BulkSubmissions(models.Model):
    """
    Stores the bulk submission directory for a course assignment.

    For each course and assignment, this table links to the directory
    containing all corresponding student submissions.
    """

    course_instance = models.ForeignKey(
        "courses.CourseInstances",
        models.CASCADE,
    )
    assignment = models.ForeignKey(
        "Assignments",
        models.CASCADE,
    )
    directory_path = models.TextField(unique=True)

    class Meta:
        """Model metadata configuration."""

        unique_together = (("course_instance", "assignment"),)
        db_table_comment = (
            "For a given course and assignment, this table stores the "
            "directory containing all student submissions."
        )

    def __str__(self):
        """
        Return a readable representation of the bulk submission.

        Includes the course instance and assignment.
        """
        return f"{self.course_instance} - {self.assignment}"


class Constraints(models.Model):
    """
    Represents constraints for an assignment.

    These include permitted or banned keywords and libraries used
    in student submissions.
    """

    assignment = models.ForeignKey(
        "Assignments",
        models.CASCADE,
    )
    identifier = models.TextField()
    is_library = models.BooleanField()
    is_keyword = models.BooleanField()
    is_permitted = models.BooleanField()

    class Meta:
        """Model metadata configuration."""

        db_table_comment = (
            "Lists the permitted/banned keywords and libraries of " "assignments."
        )

    def __str__(self):
        """
        Return a readable representation of the constraint.

        Includes status, type, identifier, and associated assignment.
        """
        status = "Permitted" if self.is_permitted else "Banned"
        type_ = "Library" if self.is_library else "Keyword"
        return f"{status} {type_}: {self.identifier} ({self.assignment})"


class PolicyViolations(models.Model):
    """
    Represents violations of assignment constraints in student submissions.

    Each entry corresponds to a detected instance where a student's code
    violated a keyword or library constraint.
    """

    constraint = models.ForeignKey(
        Constraints,
        models.CASCADE,
    )
    submission = models.ForeignKey(
        "Submissions",
        models.CASCADE,
    )
    line_number = models.BigIntegerField(blank=True, null=True)

    class Meta:
        """Model metadata configuration."""

        unique_together = (("submission", "constraint", "line_number"),)
        db_table_comment = (
            "Lists detected instances where a student's submission violates "
            "the assignment's constraints."
        )

    def __str__(self):
        """
        Return a readable representation of the policy violation.

        Includes submission, constraint, and line number.
        """
        return (
            f"Violation in {self.submission} - {self.constraint} "
            f"(Line {self.line_number})"
        )


class RequiredSubmissionFiles(models.Model):
    """
    Represents a required file for a specific assignment.

    These are the files that students must submit for CodeGrade
    grading and similarity analysis.
    """

    assignment = models.ForeignKey(
        Assignments,
        models.CASCADE,
    )
    file_name = models.TextField()
    similarity_threshold = models.DecimalField(
        max_digits=5,
        decimal_places=2,
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (("assignment", "file_name"),)
        db_table_comment = (
            "List of files that students are required to submit to "
            "CodeGrade for a given assignment."
        )

    def __str__(self):
        """
        Return a readable representation of the required file entry.

        Includes the assignment title and required file name.
        """
        return f"{self.assignment} - Required File: {self.file_name}"
