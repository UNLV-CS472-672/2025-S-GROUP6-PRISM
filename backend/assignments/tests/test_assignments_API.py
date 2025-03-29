"""Test the Assignments API endpoints."""

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from datetime import date

from assignments.models import (
    Student,
    Assignment,
    Submission,
    FlaggedSubmission,
    FlaggedStudent,
    ConfirmedCheater,
)
from courses.models import Professor, Class as CourseClass


# Helper functions to create required objects.
def create_user(
    email="test@example.com",
    password="pass123",
    first_name="Test",
    last_name="User",
):
    """Create and return a new user with the specified details.

    Args:
        email (str): The email address of the user. Defaults to "test@example.com".
        password (str): The password for the user. Defaults to "pass123".
        first_name (str): The first name of the user. Defaults to "Test".
        last_name (str): The last name of the user. Defaults to "User".

    Returns:
        User: The created user instance.
    """
    return get_user_model().objects.create_user(
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
    )


def create_professor(email="prof@example.com", first_name="Prof", last_name="One"):
    """Create and return a Professor instance with an associated User.

    Args:
        email (str): The email address for the professor's user account.
        first_name (str): The first name of the professor.
        last_name (str): The last name of the professor.

    Returns:
        Professor: A newly created Professor instance.
    """
    user = create_user(
        email=email,
        password="pass123",
        first_name=first_name,
        last_name=last_name,
    )
    return Professor.objects.create(user=user)


def create_course_class(name="Test Class"):
    """Create and return a CourseClass instance.

    Args:
        name (str): The name of the CourseClass to create.

    Returns:
        CourseClass: The created CourseClass instance.
    """
    return CourseClass.objects.create(name=name)


class StudentAPITests(APITestCase):
    """Test the Student API endpoints."""

    def setUp(self):
        """Set up test data for Student API tests."""
        self.student1 = Student.objects.create(
            email="student1@example.com",
            codeGrade_id=1001,
            username="stud1",
            first_name="Alice",
            last_name="Anderson",
        )
        self.student2 = Student.objects.create(
            email="student2@example.com",
            codeGrade_id=1002,
            username="stud2",
            first_name="Bob",
            last_name="Brown",
        )

    def test_list_students(self):
        """Test listing all students via the API."""
        url = reverse("student-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_search_students(self):
        """Test searching for students by first name."""
        url = reverse("student-list") + "?search=Alice"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["first_name"], "Alice")


class FlaggedStudentAPITests(APITestCase):
    """Test the FlaggedStudent API endpoints."""

    def setUp(self):
        """Set up test data for FlaggedStudent API tests."""
        self.prof = create_professor(
            email="prof1@example.com", first_name="Jane", last_name="Doe"
        )
        self.student = Student.objects.create(
            email="student3@example.com",
            codeGrade_id=1003,
            username="stud3",
            first_name="Charlie",
            last_name="Chaplin",
        )
        self.flagged_student = FlaggedStudent.objects.create(
            student=self.student, professor=self.prof, times_over_threshold=5
        )

    def test_list_flagged_students(self):
        """Test listing flagged students."""
        url = reverse("flagged-student-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)

    def test_filter_flagged_students(self):
        """Test filtering flagged students by professor ID."""
        url = reverse("flagged-student-list") + f"?professor__id={self.prof.id}"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["times_over_threshold"], 5)


class AssignmentAPITests(APITestCase):
    """Test the Assignment API endpoints."""

    def setUp(self):
        """Set up test data for Assignment API tests."""
        self.prof = create_professor(
            email="prof2@example.com", first_name="John", last_name="Doe"
        )
        self.course_class = create_course_class(name="Math 101")
        self.assignment1 = Assignment.objects.create(
            class_instance=self.course_class,
            professor=self.prof,
            assignment_number=1,
            title="Algebra Assignment",
            due_date=date(2023, 5, 1),
        )
        self.assignment2 = Assignment.objects.create(
            class_instance=self.course_class,
            professor=self.prof,
            assignment_number=2,
            title="Geometry Assignment",
            due_date=date(2023, 5, 15),
        )

    def test_list_assignments(self):
        """Test listing assignments via the API."""
        url = reverse("assignments-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_filter_assignments_by_title(self):
        """Test filtering assignments by title."""
        url = reverse("assignments-list") + "?title=Algebra Assignment"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["title"], "Algebra Assignment")

    def test_search_assignments(self):
        """Test searching assignments by keyword."""
        url = reverse("assignments-list") + "?search=Geometry"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["assignment_number"], 2)


class SubmissionAPITests(APITestCase):
    """Test the Submission API endpoints."""

    def setUp(self):
        """Set up test data for Submission API tests."""
        self.prof = create_professor(
            email="prof3@example.com", first_name="Alice", last_name="Smith"
        )
        self.course_class = create_course_class(name="History 101")
        self.assignment = Assignment.objects.create(
            class_instance=self.course_class,
            professor=self.prof,
            assignment_number=1,
            title="World History Assignment",
            due_date=date(2023, 6, 1),
        )
        self.student = Student.objects.create(
            email="student4@example.com",
            codeGrade_id=1004,
            username="stud4",
            first_name="David",
            last_name="Dunn",
        )
        self.submission1 = Submission.objects.create(
            student=self.student,
            assignment=self.assignment,
            grade=88,
            flagged=False,
            professor=self.prof,
        )
        self.submission2 = Submission.objects.create(
            student=self.student,
            assignment=self.assignment,
            grade=92,
            flagged=True,
            professor=self.prof,
        )

    def test_list_submissions(self):
        """Test listing all submissions."""
        url = reverse("submissions-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_filter_submissions_by_flagged(self):
        """Test filtering submissions by flagged status."""
        url = reverse("submissions-list") + "?flagged=True"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertTrue(res.data[0]["flagged"])

    def test_search_submissions(self):
        """Test searching submissions by assignment title."""
        url = reverse("submissions-list") + "?search=World History"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data[0]["assignment"]["title"], "World History Assignment")


class FlaggedSubmissionAPITests(APITestCase):
    """Test the FlaggedSubmission API endpoints."""

    def setUp(self):
        """Set up test data for FlaggedSubmission API tests."""
        self.prof = create_professor(
            email="prof4@example.com", first_name="Emma", last_name="Stone"
        )
        self.course_class = create_course_class(name="Physics 101")
        self.assignment = Assignment.objects.create(
            class_instance=self.course_class,
            professor=self.prof,
            assignment_number=1,
            title="Quantum Mechanics",
            due_date=date(2023, 7, 1),
        )
        self.student = Student.objects.create(
            email="student5@example.com",
            codeGrade_id=1005,
            username="stud5",
            first_name="Frank",
            last_name="Furter",
        )
        self.submission = Submission.objects.create(
            student=self.student,
            assignment=self.assignment,
            grade=95,
            flagged=True,
            professor=self.prof,
        )
        self.flagged_submission = FlaggedSubmission.objects.create(
            submission=self.submission,
            file_name="report.pdf",
            percentage=85,
        )
        self.flagged_submission.similarity_with.add(self.student)

    def test_list_flagged_submissions(self):
        """Test listing flagged submissions."""
        url = reverse("flagged-submission-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)

    def test_filter_flagged_submissions_by_percentage(self):
        """Test filtering flagged submissions by percentage."""
        url = reverse("flagged-submission-list") + "?percentage=85"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["percentage"], 85)

    def test_search_flagged_submissions(self):
        """Test searching flagged submissions by file name."""
        url = reverse("flagged-submission-list") + "?search=report"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data[0]["file_name"], "report.pdf")


class ConfirmedCheaterAPITests(APITestCase):
    """Test the ConfirmedCheater API endpoints."""

    def setUp(self):
        """Set up test data for ConfirmedCheater API tests."""
        self.prof = create_professor(
            email="prof5@example.com", first_name="Olivia", last_name="Newton"
        )
        self.student = Student.objects.create(
            email="student6@example.com",
            codeGrade_id=1006,
            username="stud6",
            first_name="George",
            last_name="Clooney",
        )
        self.confirmed_cheater = ConfirmedCheater.objects.create(
            student=self.student,
            professor=self.prof,
            threshold_used=45,
        )

    def test_list_confirmed_cheaters(self):
        """Test listing confirmed cheaters."""
        url = reverse("confirmed-cheater-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)

    def test_filter_confirmed_cheaters_by_threshold(self):
        """Test filtering confirmed cheaters by threshold."""
        url = reverse("confirmed-cheater-list") + "?threshold_used=45"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["threshold_used"], 45)

    def test_search_confirmed_cheaters(self):
        """Test searching confirmed cheaters by professor first name."""
        url = reverse("confirmed-cheater-list") + "?search=Olivia"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertIn("Olivia", res.data[0]["professor"]["user"]["first_name"])
