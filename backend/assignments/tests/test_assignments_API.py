"""Tests for Assignment views with filtering, ordering, search, and pagination.

This module tests the API endpoints for assignments and related models.
"""

import datetime

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.urls import reverse

from rest_framework import status
from rest_framework.test import APITestCase

from assignments.models import (
    Assignments,
    BaseFiles,
    BulkSubmissions,
    Constraints,
    PolicyViolations,
    RequiredSubmissionFiles,
    Submissions,
)
from courses.models import (
    CourseCatalog,
    CourseInstances,
    Professors,
    Semester,
    Students,
    TeachingAssistants,
)
from cheating.models import SubmissionSimilarityPairs

User = get_user_model()


class BaseViewTest(APITestCase):
    """Base test case for assignment view tests."""

    @classmethod
    def setUpTestData(cls):
        """Set up test data for view tests."""
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
        cls.ta_user = User.objects.create_user(
            username="ta1",
            password="pass123",
            email="ta1@example.com",
        )
        cls.professor = Professors.objects.create(user=cls.professor_user)
        cls.teaching_assistant = TeachingAssistants.objects.create(
            user=cls.ta_user,
        )
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
        cls.student2 = Students.objects.create(
            email="student2@example.com",
            nshe_id=12121212,
            codegrade_id=34345656,
            ace_id="ACE124",
            first_name="Jane",
            last_name="Smith",
        )
        cls.assignment = Assignments.objects.create(
            course_catalog=cls.catalog,
            semester=cls.semester,
            assignment_number=1,
            title="Test Assignment",
            due_date=datetime.date.today(),
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
        cls.sub2 = Submissions.objects.create(
            grade=90.0,
            created_at=datetime.date.today(),
            flagged=False,
            assignment=cls.assignment,
            student=cls.student2,
            course_instance=cls.course_instance,
            file_path="path/to/sub2",
        )
        cls.sim1 = SubmissionSimilarityPairs.objects.create(
            assignment=cls.assignment,
            file_name="f1.py",
            submission_id_1=cls.submission,
            submission_id_2=cls.sub2,
            match_id=1,
            percentage=80,
        )
        cls.sim2 = SubmissionSimilarityPairs.objects.create(
            assignment=cls.assignment,
            file_name="f2.py",
            submission_id_1=cls.sub2,
            submission_id_2=cls.submission,
            match_id=2,
            percentage=60,
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
        numbers = [item["assignment_number"] for item in response.data["results"]]
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
                course_catalog=self.catalog,
                semester=self.semester,
                assignment_number=100 + i,
                title=f"Assignment {100 + i}",
                due_date=datetime.date.today(),
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

    def test_get_assignments_by_course(self):
        """Test retrieving assignments via get-assignments-by-course action."""
        url = reverse("assignments-get-assignments-by-course")
        response = self.client.get(url, {"course": self.course_instance.id})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["title"], self.assignment.title)


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
        grades = [item["grade"] for item in response.data["results"]]
        self.assertEqual(grades, sorted(grades))

    def test_submissions_search(self):
        """Test searching submissions by file_path."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"search": "submission"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("submission", item["file_path"])

    def test_filter_submissions_by_multiple_params(self):
        """Test filtering submissions by assignment, course, semester, and student ID."""
        url = reverse("submissions-list")
        response = self.client.get(
            url,
            {
                "asid": self.assignment.id,
                "course": self.course_instance.id,
                "semester": self.semester.id,
                "student": self.student.id,
            },
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(response.data["count"], 1)

        # Validate the submission belongs to the right student and assignment
        submission = response.data["results"][0]
        self.assertEqual(submission["student"]["id"], self.student.id)
        self.assertEqual(submission["assignment"]["id"], self.assignment.id)
        self.assertEqual(submission["course_instance"], self.course_instance.id)

    def test_filter_submissions_by_assignment_id_only(self):
        """Test filtering submissions by assignment ID only."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"asid": self.assignment.id})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(response.data["count"], 1)
        for sub in response.data["results"]:
            self.assertEqual(sub["assignment"]["id"], self.assignment.id)

    def test_filter_submissions_by_student_id_only(self):
        """Test filtering submissions by student ID only."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"student": self.student.id})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(response.data["count"], 1)
        for sub in response.data["results"]:
            self.assertEqual(sub["student"]["id"], self.student.id)

    def test_filter_submissions_by_semester_id_only(self):
        """Test filtering submissions by semester ID only."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"semester": self.semester.id})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(response.data["count"], 1)
        for sub in response.data["results"]:
            self.assertEqual(
                sub["assignment"]["id"], self.assignment.id
            )  # same assignment implies same semester

    def test_filter_submissions_by_course_instance_id_only(self):
        """Test filtering submissions by course instance ID only."""
        url = reverse("submissions-list")
        response = self.client.get(url, {"course": self.course_instance.id})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(response.data["count"], 1)
        for sub in response.data["results"]:
            self.assertEqual(sub["course_instance"], self.course_instance.id)

    def test_filter_submissions_by_assignment_and_student(self):
        """Test filtering submissions by both assignment and student ID."""
        url = reverse("submissions-list")
        response = self.client.get(
            url, {"asid": self.assignment.id, "student": self.student2.id}
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(response.data["count"], 1)
        for sub in response.data["results"]:
            self.assertEqual(sub["assignment"]["id"], self.assignment.id)
            self.assertEqual(sub["student"]["id"], self.student2.id)


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
        names = [item["file_name"] for item in response.data["results"]]
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
        paths = [item["directory_path"] for item in response.data["results"]]
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
        identifiers = [item["identifier"] for item in response.data["results"]]
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
        lines = [item["line_number"] for item in response.data["results"]]
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
        names = [item["file_name"] for item in response.data["results"]]
        self.assertEqual(names, sorted(names))

    def test_required_submission_files_search(self):
        """Test searching required submission files by file_name."""
        url = reverse("requiredsubmissionfiles-list")
        response = self.client.get(url, {"search": "main"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("main", item["file_name"])


class AggregatedAssignmentDataViewTests(BaseViewTest):
    """Tests for the new AggregatedAssignmentDataView endpoint."""

    @classmethod
    def setUpTestData(cls):
        """Set up test data for AggregatedAssignmentDataView tests."""
        super().setUpTestData()
        cls.url = reverse(
            "aggregated-assignment-data"
        )  # make sure its in urls.py (PR 69)

    def test_anonymous_cannot_access(self):
        """Test that anonymous users cannot access the aggregation endpoint."""
        res = self.client.get(self.url)
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_professor_sees_aggregations(self):
        """Test that a professor user can retrieve aggregated assignment data."""
        # Add professor group
        prof_group, _ = Group.objects.get_or_create(name="Professor")
        self.professor_user.groups.add(prof_group)

        # Perform JWT login
        login_response = self.client.post(
            "/api/login",
            {"username": self.professor_user.email, "password": "pass123"},
            format="json",
        )
        self.assertEqual(login_response.status_code, 200, msg=login_response.json())

        # Set the JWT access cookie manually
        self.client.cookies["prism-access"] = login_response.cookies.get(
            "prism-access"
        ).value

        # Call the aggregation endpoint
        res = self.client.get(self.url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)

    def test_admin_sees_everything(self):
        """Test that an admin user can access all aggregated assignment data."""
        admin = User.objects.create_superuser(
            email="admin@example.com", password="superpass"
        )

        login_response = self.client.post(
            "/api/login",
            {"username": admin.email, "password": "superpass"},
            format="json",
        )
        self.assertEqual(login_response.status_code, 200, msg=login_response.json())

        self.client.cookies["prism-access"] = login_response.cookies.get(
            "prism-access"
        ).value

        res = self.client.get(self.url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)

    def test_filter_by_assignment(self):
        """Test filtering aggregated data by assignment ID."""
        prof_group, _ = Group.objects.get_or_create(name="Professor")
        self.professor_user.groups.add(prof_group)

        login_response = self.client.post(
            "/api/login",
            {"username": self.professor_user.email, "password": "pass123"},
            format="json",
        )
        self.assertEqual(login_response.status_code, 200, msg=login_response.json())

        self.client.cookies["prism-access"] = login_response.cookies.get(
            "prism-access"
        ).value

        res = self.client.get(self.url, {"assignments": 9999})
        self.assertEqual(res.status_code, status.HTTP_200_OK)

        data = res.json()
        for lst in data.values():
            self.assertEqual(lst, [])


class AssignmentCreationTests(BaseViewTest):
    """Test case for creating assignments with nested data."""

    def setUp(self):
        """Log in as a professor before each test."""
        prof_group, _ = Group.objects.get_or_create(name="Professor")
        self.professor_user.groups.add(prof_group)

        login_response = self.client.post(
            "/api/login",
            {"username": self.professor_user.email, "password": "pass123"},
            format="json",
        )
        self.assertEqual(login_response.status_code, 200, msg=login_response.json())
        self.client.cookies["prism-access"] = login_response.cookies.get(
            "prism-access"
        ).value

    def test_create_assignment_with_nested_data(self):
        """Test creating an assignment with base files, required files, and constraints."""
        url = reverse("assignments-list")
        payload = {
            "course_catalog": self.catalog.id,
            "semester": self.semester.id,
            "assignment_number": 2,
            "title": "Nested Test Assignment",
            "due_date": "2025-10-01",
            "lock_date": "2025-10-02T23:59:59Z",
            "pdf_filepath": "path/to/test2.pdf",
            "has_base_code": True,
            "moss_report_directory_path": "path/to/moss2",
            "bulk_ai_directory_path": "path/to/bulk2",
            "language": "Python",
            "has_policy": True,
            "base_files": [
                {"file_name": "starter.py", "file_path": "files/starter.py"},
                {"file_name": "utils.py", "file_path": "files/utils.py"},
            ],
            "required_files": [{"file_name": "main.py", "similarity_threshold": 85.0}],
            "constraints": [
                {
                    "identifier": "eval",
                    "is_library": False,
                    "is_keyword": True,
                    "is_permitted": False,
                }
            ],
        }

        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED, msg=res.json())
        self.assertEqual(res.data["title"], "Nested Test Assignment")

        # Ensure nested objects were created
        assignment_id = res.data["id"]
        self.assertTrue(BaseFiles.objects.filter(assignment_id=assignment_id).exists())
        self.assertTrue(
            RequiredSubmissionFiles.objects.filter(assignment_id=assignment_id).exists()
        )
        self.assertTrue(
            Constraints.objects.filter(assignment_id=assignment_id).exists()
        )

    def test_create_assignment_without_optional_nested_data(self):
        """Test creating an assignment with required files only (base and constraints omitted)."""
        url = reverse("assignments-list")
        payload = {
            "course_catalog": self.catalog.id,
            "semester": self.semester.id,
            "assignment_number": 3,
            "title": "Minimal Assignment",
            "due_date": "2025-10-15",
            "lock_date": "2025-10-16T23:59:59Z",
            "pdf_filepath": "path/to/basic.pdf",
            "has_base_code": False,
            "moss_report_directory_path": "path/to/moss3",
            "bulk_ai_directory_path": "path/to/bulk3",
            "language": "Java",
            "has_policy": False,
            "required_files": [{"file_name": "main.java", "similarity_threshold": 75}],
        }

        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED, msg=res.json())
        self.assertTrue(
            RequiredSubmissionFiles.objects.filter(
                assignment__title="Minimal Assignment"
            ).exists()
        )

    def test_create_assignment_with_only_base_files(self):
        """Test creating an assignment with only base files."""
        url = reverse("assignments-list")
        payload = {
            "course_catalog": self.catalog.id,
            "semester": self.semester.id,
            "assignment_number": 4,
            "title": "Assignment with Base Files",
            "due_date": "2025-10-20",
            "lock_date": "2025-10-21T23:59:59Z",
            "pdf_filepath": "files/base_only.pdf",
            "has_base_code": True,
            "moss_report_directory_path": "moss/base_only",
            "bulk_ai_directory_path": "bulk/base_only",
            "language": "Python",
            "has_policy": False,
            "base_files": [{"file_name": "setup.py", "file_path": "files/setup.py"}],
            "required_files": [{"file_name": "main.java", "similarity_threshold": 75}],
        }

        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED, msg=res.json())
        self.assertTrue(
            BaseFiles.objects.filter(
                assignment__title="Assignment with Base Files"
            ).exists()
        )

    def test_create_assignment_with_only_constraints(self):
        """Test creating an assignment with only constraints."""
        url = reverse("assignments-list")
        payload = {
            "course_catalog": self.catalog.id,
            "semester": self.semester.id,
            "assignment_number": 6,
            "title": "Assignment with Constraints",
            "due_date": "2025-10-25",
            "lock_date": "2025-10-26T23:59:59Z",
            "pdf_filepath": "files/constraints_only.pdf",
            "has_base_code": False,
            "moss_report_directory_path": "moss/constraints_only",
            "bulk_ai_directory_path": "bulk/constraints_only",
            "language": "C",
            "has_policy": True,
            "required_files": [{"file_name": "main.java", "similarity_threshold": 75}],
            "constraints": [
                {
                    "identifier": "goto",
                    "is_library": False,
                    "is_keyword": True,
                    "is_permitted": False,
                }
            ],
        }

        res = self.client.post(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED, msg=res.json())
        self.assertTrue(
            Constraints.objects.filter(
                assignment__title="Assignment with Constraints"
            ).exists()
        )
