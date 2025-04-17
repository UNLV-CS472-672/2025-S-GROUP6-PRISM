"""Test the Courses app models.

Includes coverage for string representations and unique constraints.
"""

import datetime

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

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

User = get_user_model()


class BaseCoursesTest(TestCase):
    """Base test case for the Courses app models.

    Provides common test data for all tests.
    """

    def setUp(self):
        """Set up common objects for Courses model tests."""
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
        # Provide email for the TA user as well.
        self.ta_user = User.objects.create_user(
            username="ta1",
            password="pass123",
            email="ta1@example.com",
        )
        self.professor = Professors.objects.create(user=self.professor_user)
        self.ta = TeachingAssistants.objects.create(user=self.ta_user)
        self.course_instance = CourseInstances.objects.create(
            semester=self.semester,
            course_catalog=self.catalog,
            section_number=1,
            professor=self.professor,
            teaching_assistant=self.ta,
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
        # Reflect updated Assignments model: use course_catalog & semester
        # instead of course_instance and lock_date → due_date.
        self.assignment = Assignments.objects.create(
            course_catalog=self.catalog,
            semester=self.semester,
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
        self.student_enrollment = StudentEnrollments.objects.create(
            student=self.student,
            course_instance=self.course_instance,
        )
        self.professor_enrollment = ProfessorEnrollments.objects.create(
            professor=self.professor,
            course_instance=self.course_instance,
        )
        self.ta_enrollment = TeachingAssistantEnrollments.objects.create(
            teaching_assistant=self.ta,
            course_instance=self.course_instance,
        )


class CoursesModelsStrTest(BaseCoursesTest):
    """Test string representations of Courses app models."""

    def test_course_catalog_str(self):
        """Test the __str__ method of CourseCatalog."""
        expected = (
            f"{self.catalog.subject} {self.catalog.catalog_number} "
            f"- {self.catalog.course_title}"
        )
        self.assertEqual(str(self.catalog), expected)

    def test_courses_semester_str(self):
        """Test the __str__ method of Semester."""
        expected = (
            f"{self.semester.term} {self.semester.year} - " f"{self.semester.session}"
        )
        self.assertEqual(str(self.semester), expected)

    def test_course_instances_str(self):
        """Test the __str__ method of CourseInstances."""
        expected = (
            f"{self.catalog} - Section "
            f"{self.course_instance.section_number} ({self.semester})"
        )
        self.assertEqual(str(self.course_instance), expected)

    def test_students_str(self):
        """Test the __str__ method of Students."""
        expected = (
            f"{self.student.first_name} {self.student.last_name} "
            f"({self.student.ace_id})"
        )
        self.assertEqual(str(self.student), expected)

    def test_student_enrollments_str(self):
        """Test the __str__ method of StudentEnrollments."""
        expected = f"{self.student} enrolled in " f"{self.course_instance}"
        self.assertEqual(str(self.student_enrollment), expected)

    def test_professors_str(self):
        """Test the __str__ method of Professors."""
        expected = f"Professor ID {self.professor.id} - " f"{self.professor.user}"
        self.assertEqual(str(self.professor), expected)

    def test_professor_enrollments_str(self):
        """Test the __str__ method of ProfessorEnrollments."""
        expected = f"{self.professor} assigned to " f"{self.course_instance}"
        self.assertEqual(
            str(self.professor_enrollment),
            expected,
        )

    def test_teaching_assistants_str(self):
        """Test the __str__ method of TeachingAssistants."""
        expected = f"TeachingAssistant ID {self.ta.id} - " f"{self.ta.user}"
        self.assertEqual(str(self.ta), expected)

    def test_teaching_assistant_enrollment_str(self):
        """Test the __str__ method of TeachingAssistantEnrollments."""
        expected = f"{self.ta} assigned to " f"{self.course_instance}"
        self.assertEqual(str(self.ta_enrollment), expected)


class CoursesModelsUniqueTest(BaseCoursesTest):
    """Test unique constraints for Courses app models."""

    def test_unique_course_catalog(self):
        """Test the unique constraint on (subject, catalog_number)."""
        with self.assertRaises(IntegrityError):
            CourseCatalog.objects.create(
                name="CS101-dup",
                subject=self.catalog.subject,
                catalog_number=self.catalog.catalog_number,
                course_title="Different Title",
                course_level="Undergraduate",
            )

    def test_unique_course_instances(self):
        """Test the unique constraint on CourseInstances."""
        with self.assertRaises(IntegrityError):
            CourseInstances.objects.create(
                semester=self.semester,
                course_catalog=self.catalog,
                section_number=self.course_instance.section_number,
                professor=self.professor,
                teaching_assistant=self.ta,
                canvas_course_id=654321,
            )

    def test_unique_canvas_course_id(self):
        """Test the uniqueness of canvas_course_id."""
        with self.assertRaises(IntegrityError):
            CourseInstances.objects.create(
                semester=self.semester,
                course_catalog=self.catalog,
                section_number=2,
                professor=self.professor,
                teaching_assistant=self.ta,
                canvas_course_id=(self.course_instance.canvas_course_id),
            )

    def test_unique_semester(self):
        """Test the unique constraint on (year, term, session)."""
        with self.assertRaises(IntegrityError):
            Semester.objects.create(
                name="Fall Duplicate",
                year=self.semester.year,
                term=self.semester.term,
                session=self.semester.session,
            )

    def test_unique_student_enrollments(self):
        """Test the unique constraint on StudentEnrollments."""
        with self.assertRaises(IntegrityError):
            StudentEnrollments.objects.create(
                student=self.student,
                course_instance=self.course_instance,
            )

    def test_unique_professor_enrollments(self):
        """Test the unique constraint on ProfessorEnrollments."""
        with self.assertRaises(IntegrityError):
            ProfessorEnrollments.objects.create(
                professor=self.professor,
                course_instance=self.course_instance,
            )

    def test_unique_teaching_assistant_enrollment(self):
        """Test the unique constraint on TA enrollments."""
        with self.assertRaises(IntegrityError):
            TeachingAssistantEnrollments.objects.create(
                teaching_assistant=self.ta,
                course_instance=self.course_instance,
            )
