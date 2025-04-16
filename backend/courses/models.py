"""Database models for the Courses app.

Defines core academic models: Professor, Class, Semester, ProfessorClassSection,
and Enrollment. These models manage associations between professors, students,
classes, and semesters with constraints to ensure data integrity.
"""

from django.db import models
from django.conf import settings


class CourseCatalog(models.Model):
    """
    Represents a course entry in the university's course catalog.

    Includes subject, catalog number, course title, and level.
    """

    name = models.CharField(
        unique=True,
        max_length=50,
    )
    subject = models.TextField()
    catalog_number = models.SmallIntegerField()
    course_title = models.CharField(max_length=255)
    course_level = models.TextField()

    class Meta:
        """Model metadata configuration."""

        unique_together = (("subject", "catalog_number"),)

    def __str__(self):
        """
        Return a readable representation of the catalog entry.

        Combines subject and catalog number with course title.
        """
        return f"{self.subject} {self.catalog_number} - {self.course_title}"


class CourseInstances(models.Model):
    """
    Represents a specific offering of a course in a given semester.

    Includes section number, assigned professor, optional TA, and
    Canvas course ID.
    """

    semester = models.ForeignKey(
        "Semester",
        models.CASCADE,
    )
    course_catalog = models.ForeignKey(
        CourseCatalog,
        models.CASCADE,
    )
    section_number = models.IntegerField()
    professor = models.ForeignKey(
        "Professors",
        models.CASCADE,
    )
    teaching_assistant = models.ForeignKey(
        "TeachingAssistants",
        models.CASCADE,
        blank=True,
        null=True,
    )
    canvas_course_id = models.BigIntegerField(unique=True)

    class Meta:
        """Model metadata configuration."""

        unique_together = (
            ("semester", "course_catalog", "section_number", "professor"),
        )

    def __str__(self):
        """
        Return a readable representation of the course instance.

        Includes semester, course, and section number.
        """
        return (
            f"{self.course_catalog} - Section {self.section_number} "
            f"({self.semester})"
        )


class Semester(models.Model):
    """
    Represents a semester in which courses are offered.

    Includes the academic year, term (e.g., Fall), and session type.
    """

    name = models.TextField()
    year = models.SmallIntegerField()
    term = models.TextField()
    session = models.TextField()

    class Meta:
        """Model metadata configuration."""

        unique_together = (("year", "term", "session"),)

    def __str__(self):
        """
        Return a readable representation of the semester.

        Includes term, year, and session.
        """
        return f"{self.term} {self.year} - {self.session}"


class Students(models.Model):
    """
    Represents a student in the system.

    Includes identifying information such as email, NSHE ID,
    CodeGrade ID, ACE ID, and full name.
    """

    email = models.TextField(unique=True)
    nshe_id = models.BigIntegerField(unique=True)
    codegrade_id = models.BigIntegerField(
        db_column="codeGrade_id",
        unique=True,
    )
    ace_id = models.TextField(unique=True)
    first_name = models.TextField()
    last_name = models.TextField()

    def __str__(self):
        """
        Return a readable representation of the student.

        Displays the student's full name and ACE ID.
        """
        return f"{self.first_name} {self.last_name} ({self.ace_id})"


class StudentEnrollments(models.Model):
    """
    Represents a student's enrollment in a specific course instance.

    Links a student to a course offering in a given semester.
    """

    student = models.ForeignKey(
        "Students",
        models.CASCADE,
    )
    course_instance = models.ForeignKey(
        CourseInstances,
        models.CASCADE,
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (("student", "course_instance"),)

    def __str__(self):
        """
        Return a readable representation of the enrollment.

        Displays the student and the course instance they are enrolled in.
        """
        return f"{self.student} enrolled in {self.course_instance}"


class Professors(models.Model):
    """
    Model representing a professor.

    Each professor is uniquely associated with a user account.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        models.CASCADE,
        help_text="User associated with the professor.",
    )

    def __str__(self):
        """Return a string representation of the professor."""
        return f"Professor ID {self.id} - {self.user}"


class ProfessorEnrollments(models.Model):
    """
    Represents a professor's enrollment in a course instance.

    Associates a professor with a specific course offering.
    """

    professor = models.ForeignKey(
        "Professors",
        models.CASCADE,
    )
    course_instance = models.ForeignKey(
        CourseInstances,
        models.CASCADE,
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (("professor", "course_instance"),)

    def __str__(self):
        """
        Return a readable representation of the professor's enrollment.

        Displays the professor and the course instance they are assigned to.
        """
        return f"{self.professor} assigned to {self.course_instance}"


class TeachingAssistants(models.Model):
    """
    Model representing a teaching assistant.

    Each teaching assistant is associated with one user account.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        models.CASCADE,
        help_text="User associated with the teaching assistant.",
    )

    def __str__(self):
        """Return a string representation of the teaching assistant."""
        return f"TeachingAssistant ID {self.id} - {self.user}"


class TeachingAssistantEnrollment(models.Model):
    """
    Represents a TA's enrollment in a course instance.

    Associates a teaching assistant with a specific course offering.
    """

    teaching_assistant = models.ForeignKey(
        "TeachingAssistants",
        models.CASCADE,
    )
    course_instance = models.ForeignKey(
        CourseInstances,
        models.CASCADE,
    )

    class Meta:
        """Model metadata configuration."""

        unique_together = (("teaching_assistant", "course_instance"),)

    def __str__(self):
        """
        Return a readable representation of the TA enrollment.

        Displays the teaching assistant and the course instance.
        """
        return f"{self.teaching_assistant} assigned to {self.course_instance}"
