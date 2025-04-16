"""Tests for assignments models, verifying both string representations and unique constraints.

This module tests the __str__ methods and unique constraints for the
assignments models.
"""

import datetime

from django.apps import apps
from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from assignments.models import (
    Assignments,
    Submissions,
    BaseFiles,
    BulkSubmissions,
    Constraints,
    PolicyViolations,
    RequiredSubmissionFiles,
)

User = get_user_model()

# Retrieve required models from the courses app.
Semester = apps.get_model("courses", "Semester")
CourseCatalog = apps.get_model("courses", "CourseCatalog")
Professors = apps.get_model("courses", "Professors")
TeachingAssistants = apps.get_model("courses", "TeachingAssistants")
CourseInstances = apps.get_model("courses", "CourseInstances")
Students = apps.get_model("courses", "Students")


class BaseAssignmentsTest(TestCase):
    """Base test case for assignments models.

    Creates common objects used in tests for string representations and
    unique constraints.
    """

    def setUp(self):
        """Create common objects for all assignments model tests."""
        self.semester = Semester.objects.create(
            name="Fall Semester",
            year=2025,
            term="Fall",
            session="Regular",
        )
        self.catalog = CourseCatalog.objects.create(
            name="CS101",
            subject="CS",
            catalog_number=101,
            course_title="Introduction to Computer Science",
            course_level="Undergraduate",
        )
        self.professor_user = User.objects.create_user(
            username="prof1",
            password="pass123",
            email="professor@example.com",
        )
        self.ta_user = User.objects.create_user(
            username="ta1",
            password="pass123",
            email="ta1@example.com",
        )
        self.professor = Professors.objects.create(user=self.professor_user)
        self.teaching_assistant = TeachingAssistants.objects.create(user=self.ta_user)
        self.course_instance = CourseInstances.objects.create(
            semester=self.semester,
            course_catalog=self.catalog,
            section_number=1,
            professor=self.professor,
            teaching_assistant=self.teaching_assistant,
            canvas_course_id=123456,
        )
        self.student = Students.objects.create(
            email="student@example.com",
            nshe_id=12345678,
            codegrade_id=87654321,
            ace_id="ACE123",
            first_name="John",
            last_name="Doe",
        )
        self.assignment_data = {
            "course_instance": self.course_instance,
            "assignment_number": 1,
            "title": "Test Assignment",
            "lock_date": datetime.date.today(),
            "pdf_filepath": "path/to/pdf",
            "has_base_code": True,
            "moss_report_directory_path": "path/to/moss",
            "bulk_ai_directory_path": "path/to/bulk",
            "language": "Python",
            "has_policy": True,
        }
        self.assignment = Assignments.objects.create(**self.assignment_data)


class AssignmentsModelsStrTest(BaseAssignmentsTest):
    """Tests for string representations of assignments-related models."""

    def test_assignment_str(self):
        """Check the string output of an Assignment instance."""
        expected = f"{self.course_instance} - Assignment 1: Test Assignment"
        self.assertEqual(str(self.assignment), expected)

    def test_submission_str(self):
        """Check the string output of a Submission instance."""
        submission = Submissions.objects.create(
            grade=95.0,
            created_at=datetime.date.today(),
            flagged=False,
            assignment=self.assignment,
            student=self.student,
            course_instance=self.course_instance,
            file_path="path/to/submission",
        )
        expected = f"{self.student} - {self.assignment} (Grade: 95.0)"
        self.assertEqual(str(submission), expected)

    def test_basefiles_str(self):
        """Check the string output of a BaseFiles instance."""
        base_file = BaseFiles.objects.create(
            assignment=self.assignment,
            file_name="starter.py",
            file_path="path/to/starter.py",
        )
        expected = f"{self.assignment} - starter.py"
        self.assertEqual(str(base_file), expected)

    def test_bulk_submissions_str(self):
        """Check the string output of a BulkSubmissions instance."""
        bulk_submission = BulkSubmissions.objects.create(
            course_instance=self.course_instance,
            assignment=self.assignment,
            directory_path="path/to/directory",
        )
        expected = f"{self.course_instance} - {self.assignment}"
        self.assertEqual(str(bulk_submission), expected)

    def test_constraints_str(self):
        """Check the string output of a Constraints instance."""
        constraint = Constraints.objects.create(
            assignment=self.assignment,
            identifier="forbidden_keyword",
            is_library=False,
            is_keyword=True,
            is_permitted=False,
        )
        expected_status = "Banned"
        expected_type = "Keyword"
        expected = (
            f"{expected_status} {expected_type}: forbidden_keyword "
            f"({self.assignment})"
        )
        self.assertEqual(str(constraint), expected)

    def test_policy_violations_str(self):
        """Check the string output of a PolicyViolations instance."""
        constraint = Constraints.objects.create(
            assignment=self.assignment,
            identifier="forbidden_keyword",
            is_library=False,
            is_keyword=True,
            is_permitted=False,
        )
        submission = Submissions.objects.create(
            grade=88.0,
            created_at=datetime.date.today(),
            flagged=True,
            assignment=self.assignment,
            student=self.student,
            course_instance=self.course_instance,
            file_path="path/to/flagged_submission",
        )
        policy_violation = PolicyViolations.objects.create(
            constraint=constraint,
            submission=submission,
            line_number=42,
        )
        expected = f"Violation in {submission} - {constraint} (Line 42)"
        self.assertEqual(str(policy_violation), expected)

    def test_required_submission_files_str(self):
        """Check the string output of a RequiredSubmissionFiles instance."""
        required_file = RequiredSubmissionFiles.objects.create(
            assignment=self.assignment,
            file_name="main.py",
            similarity_threshold=0.75,
        )
        expected = f"{self.assignment} - Required File: main.py"
        self.assertEqual(str(required_file), expected)


class AssignmentsModelsUniqueTest(BaseAssignmentsTest):
    """Tests for enforcing unique constraints on assignments-related models."""

    def test_unique_assignment_together(self):
        """Ensure an assignment cannot share the same course_instance and assignment_number."""
        with self.assertRaises(IntegrityError):
            Assignments.objects.create(**self.assignment_data)

    def test_unique_pdf_filepath(self):
        """Ensure that pdf_filepath is unique across Assignments."""
        new_data = self.assignment_data.copy()
        new_data["assignment_number"] = 2
        with self.assertRaises(IntegrityError):
            Assignments.objects.create(**new_data)

    def test_unique_moss_report_directory_path(self):
        """Ensure that moss_report_directory_path is unique across Assignments."""
        new_data = self.assignment_data.copy()
        new_data["assignment_number"] = 2
        new_data["pdf_filepath"] = "path/to/unique_pdf"
        with self.assertRaises(IntegrityError):
            Assignments.objects.create(**new_data)

    def test_unique_bulk_ai_directory_path(self):
        """Ensure that bulk_ai_directory_path is unique across Assignments."""
        new_data = self.assignment_data.copy()
        new_data["assignment_number"] = 2
        new_data["pdf_filepath"] = "path/to/unique_pdf2"
        new_data["moss_report_directory_path"] = "path/to/unique_moss"
        with self.assertRaises(IntegrityError):
            Assignments.objects.create(**new_data)

    def test_submission_unique_together(self):
        """Ensure Submissions cannot share the same assignment and student."""
        submission_data = {
            "grade": 90.0,
            "created_at": datetime.date.today(),
            "flagged": False,
            "assignment": self.assignment,
            "student": self.student,
            "course_instance": self.course_instance,
            "file_path": "path/to/submission1",
        }
        Submissions.objects.create(**submission_data)
        with self.assertRaises(IntegrityError):
            Submissions.objects.create(**submission_data)

    def test_basefiles_unique_together(self):
        """Ensure BaseFiles cannot share the same assignment and file_name."""
        basefile_data = {
            "assignment": self.assignment,
            "file_name": "starter.py",
            "file_path": "path/to/starter.py",
        }
        BaseFiles.objects.create(**basefile_data)
        with self.assertRaises(IntegrityError):
            BaseFiles.objects.create(**basefile_data)

    def test_bulk_submissions_unique_together(self):
        """Ensure BulkSubmissions cannot share the same course_instance and assignment."""
        bulk_data = {
            "course_instance": self.course_instance,
            "assignment": self.assignment,
            "directory_path": "path/to/directory",
        }
        BulkSubmissions.objects.create(**bulk_data)
        with self.assertRaises(IntegrityError):
            BulkSubmissions.objects.create(**bulk_data)

    def test_policy_violations_unique_together(self):
        """Ensure PolicyViolations cannot share the same submission, constraint, and line_number."""
        constraint = Constraints.objects.create(
            assignment=self.assignment,
            identifier="forbidden_keyword",
            is_library=False,
            is_keyword=True,
            is_permitted=False,
        )
        submission_data = {
            "grade": 85.0,
            "created_at": datetime.date.today(),
            "flagged": True,
            "assignment": self.assignment,
            "student": self.student,
            "course_instance": self.course_instance,
            "file_path": "path/to/submission2",
        }
        submission = Submissions.objects.create(**submission_data)
        policy_data = {
            "constraint": constraint,
            "submission": submission,
            "line_number": 10,
        }
        PolicyViolations.objects.create(**policy_data)
        with self.assertRaises(IntegrityError):
            PolicyViolations.objects.create(**policy_data)

    def test_required_submission_files_unique_together(self):
        """Ensure RequiredSubmissionFiles cannot share the same assignment and file_name."""
        req_file_data = {
            "assignment": self.assignment,
            "file_name": "main.py",
            "similarity_threshold": 0.75,
        }
        RequiredSubmissionFiles.objects.create(**req_file_data)
        with self.assertRaises(IntegrityError):
            RequiredSubmissionFiles.objects.create(**req_file_data)
