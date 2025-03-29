"""Tests for the Assignments models."""

from django.test import TestCase
from django.db.utils import IntegrityError
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model
from django.db import transaction
from datetime import date

from assignments import models as assign_models
from courses import models as course_models


def create_user(
    email="test@example.com",
    password="testpass",
    first_name="Test",
    last_name="User",
):
    """Create and return a user with the specified attributes.

    Args:
        email (str): User's email. Defaults to "test@example.com".
        password (str): User's password. Defaults to "testpass".
        first_name (str): First name. Defaults to "Test".
        last_name (str): Last name. Defaults to "User".

    Returns:
        User: The created user instance.
    """
    return get_user_model().objects.create_user(
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
    )


def create_professor(**params):
    """Create and return a Professor instance.

    Accepts custom parameters for user creation.

    Args:
        **params: Custom attributes for the user object.

    Returns:
        Professor: The created Professor instance.
    """
    defaults = {
        "email": "prof@example.com",
        "password": "testpass",
        "first_name": "Prof",
        "last_name": "One",
    }
    defaults.update(params)
    user = create_user(**defaults)
    return course_models.Professor.objects.create(user=user)


def create_class(name="Test Class"):
    """Create and return a Class instance.

    Args:
        name (str): Name of the class. Defaults to "Test Class".

    Returns:
        Class: The created Class instance.
    """
    return course_models.Class.objects.create(name=name)


class StudentModelTests(TestCase):
    """Tests for the Student model."""

    def test_student_str(self):
        """Test the string representation of a Student instance."""
        student = assign_models.Student.objects.create(
            email="student@example.com",
            codeGrade_id=1001,
            username="studentuser",
            first_name="John",
            last_name="Doe",
        )
        expected = "John Doe (student@example.com)"
        self.assertEqual(str(student), expected)

    def test_student_unique_email_and_codeGrade(self):
        """Test that duplicate emails and codeGrade_id raise IntegrityError."""
        assign_models.Student.objects.create(
            email="unique@example.com",
            codeGrade_id=2001,
            username="uniqueuser",
            first_name="Unique",
            last_name="Student",
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                assign_models.Student.objects.create(
                    email="unique@example.com",
                    codeGrade_id=2002,
                    username="anotheruser",
                    first_name="Another",
                    last_name="Student",
                )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                assign_models.Student.objects.create(
                    email="another@example.com",
                    codeGrade_id=2001,
                    username="anotheruser",
                    first_name="Another",
                    last_name="Student",
                )


class AssignmentModelTests(TestCase):
    """Tests for the Assignment model."""

    def setUp(self):
        """Set up required test data."""
        self.professor = create_professor()
        self.test_class = create_class(name="Biology 101")
        self.due_date = date(2023, 1, 1)

    def test_assignment_str(self):
        """Test the string representation of an Assignment instance."""
        assignment = assign_models.Assignment.objects.create(
            class_instance=self.test_class,
            professor=self.professor,
            assignment_number=1,
            title="Test Assignment",
            due_date=self.due_date,
        )
        expected = "Assignment 1: Test Assignment"
        self.assertEqual(str(assignment), expected)

    def test_assignment_unique_together(self):
        """Test that duplicate assignment number for the same class is invalid."""
        assign_models.Assignment.objects.create(
            class_instance=self.test_class,
            professor=self.professor,
            assignment_number=1,
            title="First Assignment",
            due_date=self.due_date,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                assign_models.Assignment.objects.create(
                    class_instance=self.test_class,
                    professor=self.professor,
                    assignment_number=1,
                    title="Duplicate Assignment Number",
                    due_date=self.due_date,
                )


class SubmissionModelTests(TestCase):
    """Tests for the Submission model."""

    def setUp(self):
        """Set up required test data."""
        self.professor = create_professor()
        self.test_class = create_class(name="Chemistry 101")
        self.due_date = date(2023, 2, 1)
        self.assignment = assign_models.Assignment.objects.create(
            class_instance=self.test_class,
            professor=self.professor,
            assignment_number=1,
            title="Chemistry Assignment",
            due_date=self.due_date,
        )
        self.student = assign_models.Student.objects.create(
            email="student2@example.com",
            codeGrade_id=3001,
            username="student2",
            first_name="Alice",
            last_name="Wonder",
        )

    def test_submission_str(self):
        """Test the string representation of a Submission instance."""
        submission = assign_models.Submission.objects.create(
            student=self.student,
            assignment=self.assignment,
            grade=85,
            flagged=False,
            professor=self.professor,
        )
        expected = f"Submission by {self.student} for {self.assignment}"
        self.assertEqual(str(submission), expected)

    def test_submission_grade_validators(self):
        """Test that invalid grade values raise a ValidationError."""
        submission = assign_models.Submission(
            student=self.student,
            assignment=self.assignment,
            grade=150,
            flagged=False,
            professor=self.professor,
        )
        with self.assertRaises(ValidationError):
            submission.full_clean()


class FlaggedSubmissionModelTests(TestCase):
    """Tests for the FlaggedSubmission model."""

    def setUp(self):
        """Set up required test data."""
        self.professor = create_professor()
        self.test_class = create_class(name="Physics 101")
        self.due_date = date(2023, 3, 1)
        self.assignment = assign_models.Assignment.objects.create(
            class_instance=self.test_class,
            professor=self.professor,
            assignment_number=1,
            title="Physics Assignment",
            due_date=self.due_date,
        )
        self.student = assign_models.Student.objects.create(
            email="student3@example.com",
            codeGrade_id=4001,
            username="student3",
            first_name="Bob",
            last_name="Builder",
        )
        self.submission = assign_models.Submission.objects.create(
            student=self.student,
            assignment=self.assignment,
            grade=90,
            flagged=True,
            professor=self.professor,
        )

    def test_flagged_submission_str(self):
        """Test the string representation of a FlaggedSubmission instance."""
        flagged = assign_models.FlaggedSubmission.objects.create(
            submission=self.submission,
            file_name="plagiarism_report.pdf",
            percentage=80,
        )
        flagged.similarity_with.add(self.student)
        expected = "Flagged Submission: plagiarism_report.pdf (80%)"
        self.assertEqual(str(flagged), expected)

    def test_flagged_submission_percentage_validators(self):
        """Test that invalid percentage values raise a ValidationError."""
        flagged = assign_models.FlaggedSubmission(
            submission=self.submission,
            file_name="low_similarity.pdf",
            percentage=10,
        )
        with self.assertRaises(ValidationError):
            flagged.full_clean()


class FlaggedStudentModelTests(TestCase):
    """Tests for the FlaggedStudent model."""

    def setUp(self):
        """Set up required test data."""
        self.professor = create_professor()
        self.student = assign_models.Student.objects.create(
            email="student4@example.com",
            codeGrade_id=5001,
            username="student4",
            first_name="Carol",
            last_name="Danvers",
        )

    def test_flagged_student_str(self):
        """Test the string representation of a FlaggedStudent instance."""
        flagged_student = assign_models.FlaggedStudent.objects.create(
            student=self.student,
            professor=self.professor,
            times_over_threshold=3,
        )
        expected = f"Flagged Student: {self.student} flagged 3 times"
        self.assertEqual(str(flagged_student), expected)


class ConfirmedCheaterModelTests(TestCase):
    """Tests for the ConfirmedCheater model."""

    def setUp(self):
        """Set up required test data."""
        self.professor = create_professor()
        self.student = assign_models.Student.objects.create(
            email="student5@example.com",
            codeGrade_id=6001,
            username="student5",
            first_name="Dave",
            last_name="Grohl",
        )

    def test_confirmed_cheater_str(self):
        """Test the string representation of a ConfirmedCheater instance."""
        confirmed = assign_models.ConfirmedCheater.objects.create(
            student=self.student, professor=self.professor, threshold_used=50
        )
        expected = (
            f"Confirmed Cheater: {self.student} "
            f"on {confirmed.confirmed_date} (Threshold: 50%)"
        )
        self.assertEqual(str(confirmed), expected)
