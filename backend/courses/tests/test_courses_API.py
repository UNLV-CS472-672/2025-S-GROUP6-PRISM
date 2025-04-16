"""Test the Courses API endpoints.

This file tests the viewsets for the Courses app for filtering,
ordering, search, and pagination functionality.
"""

import datetime

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from courses.models import (
    CourseCatalog,
    CourseInstances,
    Semester,
    Students,
    StudentEnrollments,
    Professors,
    ProfessorEnrollments,
    TeachingAssistants,
    TeachingAssistantEnrollments,
)
from assignments.models import Assignments
from django.contrib.auth import get_user_model

User = get_user_model()


class BaseCoursesAPITest(APITestCase):
    """Base test case for Courses API tests.

    Sets up common objects required for API endpoint tests.
    """

    @classmethod
    def setUpTestData(cls):
        """Set up test data for the Courses API tests."""
        cls.semester = Semester.objects.create(
            name="Fall Semester",
            year=2025,
            term="Fall",
            session="Regular",
        )
        cls.catalog = CourseCatalog.objects.create(
            name="CS101",
            subject="CS",
            catalog_number=101,
            course_title="Intro to CS",
            course_level="Undergraduate",
        )
        cls.professor_user = User.objects.create_user(
            username="prof1",
            password="pass123",
            email="professor@example.com",
        )
        # Provide email for TA as well.
        cls.ta_user = User.objects.create_user(
            username="ta1",
            password="pass123",
            email="ta1@example.com",
        )
        cls.professor = Professors.objects.create(user=cls.professor_user)
        cls.ta = TeachingAssistants.objects.create(user=cls.ta_user)
        cls.course_instance = CourseInstances.objects.create(
            semester=cls.semester,
            course_catalog=cls.catalog,
            section_number=1,
            professor=cls.professor,
            teaching_assistant=cls.ta,
            canvas_course_id=123456,
        )
        cls.student = Students.objects.create(
            email="student@example.com",
            nshe_id=12345678,
            codegrade_id=87654321,
            ace_id="ACE123",
            first_name="John",
            last_name="Doe",
        )
        cls.assignment = Assignments.objects.create(
            course_instance=cls.course_instance,
            assignment_number=1,
            title="Test Assignment",
            lock_date=datetime.date.today(),
            pdf_filepath="path/to/pdf",
            has_base_code=True,
            moss_report_directory_path="path/to/moss",
            bulk_ai_directory_path="path/to/bulk",
            language="Python",
            has_policy=True,
        )
        cls.student_enrollment = StudentEnrollments.objects.create(
            student=cls.student,
            course_instance=cls.course_instance,
        )
        cls.professor_enrollment = ProfessorEnrollments.objects.create(
            professor=cls.professor,
            course_instance=cls.course_instance,
        )
        cls.ta_enrollment = TeachingAssistantEnrollments.objects.create(
            teaching_assistant=cls.ta,
            course_instance=cls.course_instance,
        )


class CourseCatalogAPITest(BaseCoursesAPITest):
    """Test API endpoints for CourseCatalogViewSet."""

    def test_coursecatalog_list(self):
        """Test retrieving a list of course catalog entries."""
        url = reverse("coursecatalog-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_coursecatalog_filter(self):
        """Test filtering course catalog entries by subject."""
        url = reverse("coursecatalog-list")
        response = self.client.get(url, {"subject": self.catalog.subject})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["subject"], self.catalog.subject)

    def test_coursecatalog_ordering(self):
        """Test ordering course catalog entries by catalog_number."""
        url = reverse("coursecatalog-list")
        response = self.client.get(url, {"ordering": "catalog_number"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        numbers = [item["catalog_number"] for item in response.data["results"]]
        self.assertEqual(numbers, sorted(numbers))

    def test_coursecatalog_search(self):
        """Test searching course catalog entries by course_title."""
        url = reverse("coursecatalog-list")
        response = self.client.get(url, {"search": "Intro"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("Intro", item["course_title"])


class CourseInstancesAPITest(BaseCoursesAPITest):
    """Test API endpoints for CourseInstancesViewSet."""

    def test_courseinstances_list(self):
        """Test retrieving a list of course instances."""
        url = reverse("courseinstances-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_courseinstances_filter(self):
        """Test filtering course instances by section_number."""
        url = reverse("courseinstances-list")
        response = self.client.get(
            url, {"section_number": self.course_instance.section_number}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(
                item["section_number"], self.course_instance.section_number
            )

    def test_courseinstances_ordering(self):
        """Test ordering course instances by section_number."""
        url = reverse("courseinstances-list")
        response = self.client.get(url, {"ordering": "section_number"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        sections = [item["section_number"] for item in response.data["results"]]
        self.assertEqual(sections, sorted(sections))

    def test_courseinstances_search(self):
        """Test searching course instances by course_catalog__course_title."""
        url = reverse("courseinstances-list")
        response = self.client.get(url, {"search": "Intro"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Since the serializer returns the primary key for course_catalog,
        # we check that at least one result has the expected course_catalog ID.
        found = any(
            item.get("course_catalog") == self.catalog.pk
            for item in response.data["results"]
        )
        self.assertTrue(found)


class SemesterAPITest(BaseCoursesAPITest):
    """Test API endpoints for SemesterViewSet."""

    def test_semester_list(self):
        """Test retrieving a list of courses semester entries."""
        url = reverse("semester-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_semester_filter(self):
        """Test filtering courses semester entries by year."""
        url = reverse("semester-list")
        response = self.client.get(url, {"year": self.semester.year})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["year"], self.semester.year)

    def test_semester_ordering(self):
        """Test ordering courses semester entries by year."""
        url = reverse("semester-list")
        response = self.client.get(url, {"ordering": "year"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        years = [item["year"] for item in response.data["results"]]
        self.assertEqual(years, sorted(years))

    def test_semester_search(self):
        """Test searching courses semester entries by term."""
        url = reverse("semester-list")
        response = self.client.get(url, {"search": self.semester.term})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn(self.semester.term, item["term"])


class StudentsAPITest(BaseCoursesAPITest):
    """Test API endpoints for StudentsViewSet."""

    def test_students_list(self):
        """Test retrieving a list of students."""
        url = reverse("students-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_students_filter(self):
        """Test filtering students by email."""
        url = reverse("students-list")
        response = self.client.get(url, {"email": self.student.email})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["email"], self.student.email)

    def test_students_ordering(self):
        """Test ordering students by first_name."""
        url = reverse("students-list")
        response = self.client.get(url, {"ordering": "first_name"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        names = [item["first_name"] for item in response.data["results"]]
        self.assertEqual(names, sorted(names))

    def test_students_search(self):
        """Test searching students by last_name."""
        url = reverse("students-list")
        response = self.client.get(url, {"search": self.student.last_name})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn(self.student.last_name, item["last_name"])


class StudentEnrollmentsAPITest(BaseCoursesAPITest):
    """Test API endpoints for StudentEnrollmentsViewSet."""

    def test_studentenrollments_list(self):
        """Test retrieving a list of student enrollments."""
        url = reverse("studentenrollments-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)


class ProfessorsAPITest(BaseCoursesAPITest):
    """Test API endpoints for ProfessorsViewSet."""

    def test_professors_list(self):
        """Test retrieving a list of professors."""
        url = reverse("professors-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_professors_filter(self):
        """Test filtering professors by user ID."""
        url = reverse("professors-list")
        response = self.client.get(url, {"user": self.professor.user.id})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["user"], self.professor.user.id)

    def test_professors_search(self):
        """Test searching professors by username."""
        url = reverse("professors-list")
        response = self.client.get(url, {"search": self.professor.user.username})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Since the default serializer only returns the user ID (e.g., "user": 1),
        # we verify that at least one result has the same user ID as our professor.
        found = any(
            item.get("user") == self.professor.user.id
            for item in response.data["results"]
        )
        self.assertTrue(found)


class ProfessorEnrollmentsAPITest(BaseCoursesAPITest):
    """Test API endpoints for ProfessorEnrollmentsViewSet."""

    def test_professorenrollments_list(self):
        """Test retrieving a list of professor enrollments."""
        url = reverse("professorenrollments-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)


class TeachingAssistantsAPITest(BaseCoursesAPITest):
    """Test API endpoints for TeachingAssistantsViewSet."""

    def test_teachingassistants_list(self):
        """Test retrieving a list of teaching assistants."""
        url = reverse("teachingassistants-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_teachingassistants_filter(self):
        """Test filtering teaching assistants by user ID."""
        url = reverse("teachingassistants-list")
        response = self.client.get(url, {"user": self.ta.user.id})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["user"], self.ta.user.id)

    def test_teachingassistants_search(self):
        """Test searching teaching assistants by username."""
        url = reverse("teachingassistants-list")
        response = self.client.get(url, {"search": self.ta.user.username})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Since the serializer returns the user ID (e.g., "user": 1) rather than
        # a nested "user__username" field, check that one of the results has the
        # expected user ID.
        found = any(
            item.get("user") == self.ta.user.id for item in response.data["results"]
        )
        self.assertTrue(found)


class TeachingAssistantEnrollmentsAPITest(BaseCoursesAPITest):
    """Test API endpoints for TeachingAssistantEnrollmentsViewSet."""

    def test_teachingassistantenrollments_list(self):
        """Test retrieving a list of teaching assistant enrollments."""
        url = reverse("teachingassistantenrollments-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
