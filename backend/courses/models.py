"""Database models for the Courses app.

Defines core academic models: Professor, Class, Semester, ProfessorClassSection,
and Enrollment. These models manage associations between professors, students,
classes, and semesters with constraints to ensure data integrity.
"""

from django.db import models
from users import models as user_model
from assignments.models import Student


class Professor(models.Model):
    """Represent a professor in the system.

    Attributes:
        user (OneToOneField): Link to the User model.

    Methods:
        __str__(): Return full name of the professor.
    """

    user = models.OneToOneField(
        user_model.User,
        on_delete=models.CASCADE,
        related_name="user_professor",
    )

    def __str__(self):
        """Return full name of the professor."""
        return f"{self.user.first_name} {self.user.last_name}"


class Class(models.Model):
    """Represent a class or course in the system.

    Attributes:
        name (CharField): Unique name of the class.
        professors (ManyToManyField): Relationship to Professor via intermediate model.

    Methods:
        __str__(): Return the class name.
    """

    name = models.CharField(max_length=50, unique=True)
    professors = models.ManyToManyField(
        "courses.Professor",
        through="ProfessorClassSection",
    )

    def __str__(self):
        """Return the class name."""
        return self.name


class Semester(models.Model):
    """Represent a semester in the system.

    Attributes:
        name (CharField): Unique name of the semester.

    Methods:
        __str__(): Return the semester name.
    """

    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        """Return the semester name."""
        return self.name


class ProfessorClassSection(models.Model):
    """Map a professor to a class section in a specific semester.

    Attributes:
        professor (ForeignKey): Link to Professor.
        class_instance (ForeignKey): Link to Class.
        semester (ForeignKey): Link to Semester.
        section_number (IntegerField): Optional section number.

    Methods:
        __str__(): Return formatted section details.
    """

    professor = models.ForeignKey(
        "courses.Professor",
        on_delete=models.CASCADE,
        related_name="profclassect",
    )
    class_instance = models.ForeignKey(Class, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)
    section_number = models.IntegerField(blank=True, null=True)

    def __str__(self):
        """Return a formatted string for this section."""
        return (
            f"{self.professor} - {self.class_instance} - "
            f"{self.semester} (Section {self.section_number})"
        )


class Enrollment(models.Model):
    """Track student enrollment status in classes.

    Attributes:
        student (ForeignKey): Link to Student.
        class_instance (ForeignKey): Link to Class.
        semester (ForeignKey): Link to Semester.
        enrolled_date (DateField): Auto-set on creation.
        dropped (BooleanField): Flag for drop status.
        dropped_date (DateField): Optional timestamp of when they dropped.

    Methods:
        __str__(): Return enrollment summary.
    """

    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    class_instance = models.ForeignKey(Class, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)
    enrolled_date = models.DateField(auto_now_add=True)
    dropped = models.BooleanField(default=False)
    dropped_date = models.DateField(blank=True, null=True)

    class Meta:
        """Meta options for Enrollment model."""

        unique_together = ("student", "class_instance", "semester")

    def __str__(self):
        """Return formatted enrollment string."""
        return f"{
            self.student} enrolled in {
            self.class_instance} ({
            self.semester})"
