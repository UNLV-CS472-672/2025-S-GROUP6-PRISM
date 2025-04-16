"""Tests for Assignment views with filtering, ordering, search, and pagination.

This module tests the API endpoints for assignments and related models.
"""

import datetime

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from assignments.models import (
    Assignments,
    Submissions,
    BaseFiles,
    BulkSubmissions,
    Constraints,
    PolicyViolations,
    RequiredSubmissionFiles,
)
from courses.models import (
    Semester,
    CourseCatalog,
    CourseInstances,
    Professors,
    TeachingAssistants,
    Students,
)
from django.contrib.auth import get_user_model

User = get_user_model()


class BaseViewTest(APITestCase):
    """Base test case for assignment view tests."""

    @classmethod
    def setUpTestData(cls):
        """Set up test data for view tests."""
        cls.semester = Semester.objects.create(
            name="Fall Semester", year=2025, term="Fall", session="Regular"
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
        cls.ta_user = User.objects.create_user(
            username="ta1",
            password="pass123",
            email="ta1@example.com",
        )
        cls.professor = Professors.objects.create(user=cls.professor_user)
        cls.teaching_assistant = TeachingAssistants.objects.create(user=cls.ta_user)
        cls.course_instance = CourseInstances.objects.create(
            semester=cls.semester,
            course_catalog=cls.catalog,
            section_number=1,
            professor=cls.professor,
            teaching_assistant=cls.teaching_assistant,
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
        cls.submission = Submissions.objects.create(
            grade=95.0,
            created_at=datetime.date.today(),
            flagged=False,
            assignment=cls.assignment,
            student=cls.student,
            course_instance=cls.course_instance,
            file_path="path/to/submission",
        )
        cls.base_file = BaseFiles.objects.create(
            assignment=cls.assignment,
            file_name="starter.py",
            file_path="path/to/starter.py",
        )
        cls.bulk_submission = BulkSubmissions.objects.create(
            course_instance=cls.course_instance,
            assignment=cls.assignment,
            directory_path="path/to/directory",
        )
        cls.constraint = Constraints.objects.create(
            assignment=cls.assignment,
            identifier="forbidden_keyword",
            is_library=False,
            is_keyword=True,
            is_permitted=False,
        )
        cls.policy_violation = PolicyViolations.objects.create(
            constraint=cls.constraint,
            submission=cls.submission,
            line_number=42,
        )
        cls.req_sub_file = RequiredSubmissionFiles.objects.create(
            assignment=cls.assignment,
            file_name="main.py",
            similarity_threshold=0.75,
        )


class AssignmentsViewSetTest(BaseViewTest):
    """Tests for the AssignmentsViewSet endpoints."""

    def test_assignments_list(self):
        """Test retrieving the list of assignments."""
        url = reverse("assignments-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_assignments_filter(self):
        """Test filtering assignments by assignment_number."""
        url = reverse("assignments-list")
        response = self.client.get(url, {"assignment_number": 1})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["assignment_number"], 1)

    def test_assignments_ordering(self):
        """Test ordering assignments by assignment_number."""
        url = reverse("assignments-list")
        response = self.client.get(url, {"ordering": "assignment_number"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = response.data["results"]
        numbers = [item["assignment_number"] for item in results]
        self.assertEqual(numbers, sorted(numbers))

    def test_assignments_search(self):
        """Test searching assignments by title."""
        url = reverse("assignments-list")
        response = self.client.get(url, {"search": "Test Assignment"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("Test Assignment", item["title"])

    def test_assignments_pagination(self):
        """Test pagination on the assignments endpoint."""
        for i in range(15):
            Assignments.objects.create(
                course_instance=self.course_instance,
                assignment_number=100 + i,
                title=f"Assignment {100 + i}",
                lock_date=datetime.date.today(),
                pdf_filepath=f"path/to/pdf{i}",
                has_base_code=True,
                moss_report_directory_path=f"path/to/moss{i}",
                bulk_ai_directory_path=f"path/to/bulk{i}",
                language="Python",
                has_policy=True,
            )
        url = reverse("assignments-list")
        response = self.client.get(url, {"page_size": 10})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 10)
        self.assertIn("next", response.data)
        self.assertIn("previous", response.data)


class SubmissionsViewSetTest(BaseViewTest):
    """Tests for the SubmissionsViewSet endpoints."""

    def test_submissions_list(self):
        """Test retrieving the list of submissions."""
        url = reverse("submissions-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_submissions_filter(self):
        """Test filtering submissions by flagged status."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"flagged": False})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertFalse(item["flagged"])

    def test_submissions_ordering(self):
        """Test ordering submissions by grade."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"ordering": "grade"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = response.data["results"]
        grades = [item["grade"] for item in results]
        self.assertEqual(grades, sorted(grades))

    def test_submissions_search(self):
        """Test searching submissions by file_path."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"search": "submission"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("submission", item["file_path"])


class BaseFilesViewSetTest(BaseViewTest):
    """Tests for the BaseFilesViewSet endpoints."""

    def test_basefiles_list(self):
        """Test retrieving the list of base files."""
        url = reverse("basefiles-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_basefiles_filter(self):
        """Test filtering base files by file_name."""
        url = reverse("basefiles-list")
        response = self.client.get(url, {"file_name": "starter.py"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["file_name"], "starter.py")

    def test_basefiles_ordering(self):
        """Test ordering base files by file_name."""
        url = reverse("basefiles-list")
        response = self.client.get(url, {"ordering": "file_name"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = response.data["results"]
        names = [item["file_name"] for item in results]
        self.assertEqual(names, sorted(names))

    def test_basefiles_search(self):
        """Test searching base files by file_path."""
        url = reverse("basefiles-list")
        response = self.client.get(url, {"search": "starter"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("starter", item["file_path"])


class BulkSubmissionsViewSetTest(BaseViewTest):
    """Tests for the BulkSubmissionsViewSet endpoints."""

    def test_bulk_submissions_list(self):
        """Test retrieving the list of bulk submissions."""
        url = reverse("bulksubmissions-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_bulk_submissions_filter(self):
        """Test filtering bulk submissions by directory_path."""
        url = reverse("bulksubmissions-list")
        response = self.client.get(url, {"directory_path": "path/to/directory"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["directory_path"], "path/to/directory")

    def test_bulk_submissions_ordering(self):
        """Test ordering bulk submissions by directory_path."""
        url = reverse("bulksubmissions-list")
        response = self.client.get(url, {"ordering": "directory_path"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = response.data["results"]
        paths = [item["directory_path"] for item in results]
        self.assertEqual(paths, sorted(paths))

    def test_bulk_submissions_search(self):
        """Test searching bulk submissions by directory_path."""
        url = reverse("bulksubmissions-list")
        response = self.client.get(url, {"search": "directory"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("directory", item["directory_path"])


class ConstraintsViewSetTest(BaseViewTest):
    """Tests for the ConstraintsViewSet endpoints."""

    def test_constraints_list(self):
        """Test retrieving the list of constraints."""
        url = reverse("constraints-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_constraints_filter(self):
        """Test filtering constraints by identifier."""
        url = reverse("constraints-list")
        response = self.client.get(url, {"identifier": "forbidden_keyword"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["identifier"], "forbidden_keyword")

    def test_constraints_ordering(self):
        """Test ordering constraints by identifier."""
        url = reverse("constraints-list")
        response = self.client.get(url, {"ordering": "identifier"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = response.data["results"]
        identifiers = [item["identifier"] for item in results]
        self.assertEqual(identifiers, sorted(identifiers))

    def test_constraints_search(self):
        """Test searching constraints by identifier."""
        url = reverse("constraints-list")
        response = self.client.get(url, {"search": "forbidden"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("forbidden", item["identifier"])


class PolicyViolationsViewSetTest(BaseViewTest):
    """Tests for the PolicyViolationsViewSet endpoints."""

    def test_policy_violations_list(self):
        """Test retrieving the list of policy violations."""
        url = reverse("policyviolations-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_policy_violations_filter(self):
        """Test filtering policy violations by line_number."""
        url = reverse("policyviolations-list")
        response = self.client.get(url, {"line_number": 42})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["line_number"], 42)

    def test_policy_violations_ordering(self):
        """Test ordering policy violations by line_number."""
        url = reverse("policyviolations-list")
        response = self.client.get(url, {"ordering": "line_number"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = response.data["results"]
        lines = [item["line_number"] for item in results]
        self.assertEqual(lines, sorted(lines))

    def test_policy_violations_search(self):
        """Test that search does not filter policy violations."""
        url = reverse("policyviolations-list")
        response = self.client.get(url, {"search": "anything"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data["results"]), 1)


class RequiredSubmissionFilesViewSetTest(BaseViewTest):
    """Tests for the RequiredSubmissionFilesViewSet endpoints."""

    def test_required_submission_files_list(self):
        """Test retrieving the list of required submission files."""
        url = reverse("requiredsubmissionfiles-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_required_submission_files_filter(self):
        """Test filtering required submission files by file_name."""
        url = reverse("requiredsubmissionfiles-list")
        response = self.client.get(url, {"file_name": "main.py"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["file_name"], "main.py")

    def test_required_submission_files_ordering(self):
        """Test ordering required submission files by file_name."""
        url = reverse("requiredsubmissionfiles-list")
        response = self.client.get(url, {"ordering": "file_name"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = response.data["results"]
        names = [item["file_name"] for item in results]
        self.assertEqual(names, sorted(names))

    def test_required_submission_files_search(self):
        """Test searching required submission files by file_name."""
        url = reverse("requiredsubmissionfiles-list")
        response = self.client.get(url, {"search": "main"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("main", item["file_name"])
