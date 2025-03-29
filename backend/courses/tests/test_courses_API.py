"""Tests for the Courses API endpoints."""

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model

from courses.models import Professor, Semester, Class, ProfessorClassSection


def create_user(
    email="test@example.com", password="pass123", first_name="Test", last_name="User"
):
    """Create and return a new user.

    Args:
        email (str): The email address of the user.
        password (str): The password for the user.
        first_name (str): The first name of the user.
        last_name (str): The last name of the user.

    Returns:
        User: The created user instance.
    """
    return get_user_model().objects.create_user(
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
    )


def create_professor(email, first_name, last_name):
    """Create and return a Professor instance.

    Args:
        email (str): The email address of the professor.
        first_name (str): The first name of the professor.
        last_name (str): The last name of the professor.

    Returns:
        Professor: The created Professor instance.
    """
    user = create_user(
        email=email,
        password="pass123",
        first_name=first_name,
        last_name=last_name,
    )
    return Professor.objects.create(user=user)


class ProfessorAPITests(APITestCase):
    """Test suite for the Professor API endpoints."""

    def setUp(self):
        """Set up the test environment."""
        self.prof1 = create_professor("john@example.com", "John", "Doe")
        self.prof2 = create_professor("jane@example.com", "Jane", "Smith")

    def test_list_professors(self):
        """Test retrieving a list of professors."""
        url = reverse("professor-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_filter_professors_by_first_name(self):
        """Test filtering professors by first name."""
        url = reverse("professor-list") + "?user__first_name=Jane"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["user"]["first_name"], "Jane")

    def test_ordering_professors(self):
        """Test ordering professors by first name descending."""
        url = reverse("professor-list") + "?ordering=-user__first_name"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        first_names = [item["user"]["first_name"] for item in res.data]
        self.assertEqual(first_names, sorted(first_names, reverse=True))

    def test_search_professors(self):
        """Test searching professors by first name."""
        url = reverse("professor-list") + "?search=John"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertTrue(any("John" in item["user"]["first_name"] for item in res.data))


class SemesterAPITests(APITestCase):
    """Test suite for the Semester API endpoints."""

    def setUp(self):
        """Set up the test environment."""
        self.sem1 = Semester.objects.create(name="Fall 2023")
        self.sem2 = Semester.objects.create(name="Spring 2023")

    def test_list_semesters(self):
        """Test retrieving a list of semesters."""
        url = reverse("semester-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_filter_semesters_by_name(self):
        """Test filtering semesters by name."""
        url = reverse("semester-list") + "?name=Fall 2023"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["name"], "Fall 2023")

    def test_search_semesters(self):
        """Test searching semesters by name."""
        url = reverse("semester-list") + "?search=Spring"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data[0]["name"], "Spring 2023")


class ClassAPITests(APITestCase):
    """Test suite for the Class API endpoints."""

    def setUp(self):
        """Set up the test environment."""
        self.class1 = Class.objects.create(name="Math 101")
        self.class2 = Class.objects.create(name="History 101")

    def test_list_classes(self):
        """Test retrieving a list of classes."""
        url = reverse("class-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_search_classes(self):
        """Test searching classes by name."""
        url = reverse("class-list") + "?search=Math"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["name"], "Math 101")


class ProfessorClassSectionAPITests(APITestCase):
    """Test suite for the ProfessorClassSection API."""

    def setUp(self):
        """Set up the test environment."""
        self.prof1 = create_professor("alice@example.com", "Alice", "Wonderland")
        self.prof2 = create_professor("bob@example.com", "Bob", "Builder")
        self.semester = Semester.objects.create(name="Fall 2023")
        self.class_obj = Class.objects.create(name="Physics 101")
        self.section1 = ProfessorClassSection.objects.create(
            professor=self.prof1,
            class_instance=self.class_obj,
            semester=self.semester,
            section_number=1,
        )
        self.section2 = ProfessorClassSection.objects.create(
            professor=self.prof2,
            class_instance=self.class_obj,
            semester=self.semester,
            section_number=2,
        )

    def test_list_professor_class_sections(self):
        """Test retrieving a list of professor class sections."""
        url = reverse("sectionclassprof-list")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_filter_by_semester_name(self):
        """Test filtering professor class sections by semester name."""
        url = reverse("sectionclassprof-list") + "?semester__name=Fall 2023"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

    def test_search_professor_class_sections(self):
        """Test searching professor class sections by class name."""
        url = reverse("sectionclassprof-list") + "?search=Physics"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)
