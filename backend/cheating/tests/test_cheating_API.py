"""Test the Cheating API endpoints.

This file tests the viewsets for the Cheating app for filtering, ordering,
search, and pagination functionality.
"""

import datetime

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from assignments.models import Assignments
from cheating.models import (
    CheatingGroups,
    CheatingGroupMembers,
    ConfirmedCheaters,
    LongitudinalCheatingGroups,
    LongitudinalCheatingGroupMembers,
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


class BaseCheatingAPITest(APITestCase):
    """Base test case for Cheating API tests.

    Sets up common objects required for the Cheating API endpoints.
    """

    @classmethod
    def setUpTestData(cls):
        """Set up test data for Cheating API tests."""
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


class CheatingGroupsAPITest(BaseCheatingAPITest):
    """Tests for the CheatingGroupsViewSet endpoints."""

    def test_cheating_groups_list(self):
        """Test retrieving a list of cheating groups."""
        CheatingGroups.objects.create(
            assignment=self.assignment,
            cohesion_score=0.85,
            analysis_report_path="path/to/report1",
        )
        url = reverse("cheating-groups-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)
        self.assertGreaterEqual(len(response.data["results"]), 1)

    def test_cheating_groups_filter(self):
        """Test filtering cheating groups by cohesion_score."""
        CheatingGroups.objects.create(
            assignment=self.assignment,
            cohesion_score=0.90,
            analysis_report_path="path/to/report2",
        )
        url = reverse("cheating-groups-list")
        response = self.client.get(url, {"cohesion_score": 0.90})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["cohesion_score"], 0.90)

    def test_cheating_groups_ordering(self):
        """Test ordering cheating groups by cohesion_score."""
        CheatingGroups.objects.create(
            assignment=self.assignment,
            cohesion_score=0.70,
            analysis_report_path="path/to/report3",
        )
        url = reverse("cheating-groups-list")
        response = self.client.get(url, {"ordering": "cohesion_score"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        scores = [item["cohesion_score"] for item in response.data["results"]]
        self.assertEqual(scores, sorted(scores))

    def test_cheating_groups_search(self):
        """Test searching cheating groups by analysis_report_path."""
        CheatingGroups.objects.create(
            assignment=self.assignment,
            cohesion_score=0.95,
            analysis_report_path="unique_report_path",
        )
        url = reverse("cheating-groups-list")
        response = self.client.get(url, {"search": "unique_report_path"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        found = any(
            "unique_report_path" in item["analysis_report_path"]
            for item in response.data["results"]
        )
        self.assertTrue(found)


class CheatingGroupMembersAPITest(BaseCheatingAPITest):
    """Tests for the CheatingGroupMembersViewSet endpoints."""

    def test_cheating_group_members_list(self):
        """Test retrieving a list of cheating group members."""
        group = CheatingGroups.objects.create(
            assignment=self.assignment,
            cohesion_score=0.90,
            analysis_report_path="path/to/report4",
        )
        CheatingGroupMembers.objects.create(
            cheating_group=group,
            student=self.student,
            cluster_distance=0.15,
        )
        url = reverse("cheating-group-members-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_cheating_group_members_ordering(self):
        """Test ordering cheating group members by cluster_distance."""
        group = CheatingGroups.objects.create(
            assignment=self.assignment,
            cohesion_score=0.92,
            analysis_report_path="path/to/report5",
        )
        CheatingGroupMembers.objects.create(
            cheating_group=group,
            student=self.student,
            cluster_distance=0.20,
        )
        CheatingGroupMembers.objects.create(
            cheating_group=group,
            student=self.student,
            cluster_distance=0.10,
        )
        url = reverse("cheating-group-members-list")
        response = self.client.get(url, {"ordering": "cluster_distance"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        distances = [item["cluster_distance"] for item in response.data["results"]]
        self.assertEqual(distances, sorted(distances))


class ConfirmedCheatersAPITest(BaseCheatingAPITest):
    """Tests for the ConfirmedCheatersViewSet endpoints."""

    def test_confirmed_cheaters_list(self):
        """Test retrieving a list of confirmed cheaters."""
        confirmed_date = datetime.date.today()
        ConfirmedCheaters.objects.create(
            confirmed_date=confirmed_date,
            threshold_used=80,
            assignment=self.assignment,
            student=self.student,
        )
        url = reverse("confirmed-cheaters-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_confirmed_cheaters_filter(self):
        """Test filtering confirmed cheaters by confirmed_date."""
        confirmed_date = datetime.date.today()
        ConfirmedCheaters.objects.create(
            confirmed_date=confirmed_date,
            threshold_used=80,
            assignment=self.assignment,
            student=self.student,
        )
        url = reverse("confirmed-cheaters-list")
        response = self.client.get(url, {"confirmed_date": confirmed_date})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            # Expecting a string in YYYY-MM-DD format.
            self.assertEqual(
                item["confirmed_date"], confirmed_date.strftime("%Y-%m-%d")
            )


class SubmissionSimiliarityPairsAPITest(BaseCheatingAPITest):
    """Tests for the SubmissionSimiliarityPairsViewSet endpoints."""

    def test_submission_similarity_pairs_search(self):
        """Test searching submission similarity pairs by file_name."""
        url = reverse("submission-similarity-pairs-list")
        response = self.client.get(url, {"search": "code.py"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertIn("code.py", item["file_name"])


class LongitudinalCheatingGroupsAPITest(BaseCheatingAPITest):
    """Tests for the LongitudinalCheatingGroupsViewSet endpoints."""

    def test_longitudinal_cheating_groups_list(self):
        """Test retrieving a list of longitudinal cheating groups."""
        LongitudinalCheatingGroups.objects.create(score=75.5)
        url = reverse("longitudinal-cheating-groups-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_longitudinal_cheating_groups_filter(self):
        """Test filtering longitudinal cheating groups by score."""
        LongitudinalCheatingGroups.objects.create(score=88.0)
        url = reverse("longitudinal-cheating-groups-list")
        response = self.client.get(url, {"score": 88.0})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for item in response.data["results"]:
            self.assertEqual(item["score"], 88.0)

    def test_longitudinal_cheating_groups_ordering(self):
        """Test ordering longitudinal cheating groups by score."""
        LongitudinalCheatingGroups.objects.create(score=65.0)
        url = reverse("longitudinal-cheating-groups-list")
        response = self.client.get(url, {"ordering": "score"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        scores = [item["score"] for item in response.data["results"]]
        self.assertEqual(scores, sorted(scores))


class LongitudinalCheatingGroupMembersAPITest(BaseCheatingAPITest):
    """Tests for the LongitudinalCheatingGroupMembersViewSet endpoints."""

    def test_longitudinal_cheating_group_members_list(self):
        """Test retrieving a list of longitudinal cheating group members."""
        long_group = LongitudinalCheatingGroups.objects.create(score=80.0)
        LongitudinalCheatingGroupMembers.objects.create(
            longitudinal_cheating_group=long_group,
            student=self.student,
            is_core_member=True,
            appearance_count=3,
        )
        url = reverse("longitudinal-cheating-group-members-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("results", response.data)

    def test_longitudinal_cheating_group_members_ordering(self):
        """Test ordering longitudinal cheating group members by appearance_count."""
        long_group = LongitudinalCheatingGroups.objects.create(score=85.0)
        LongitudinalCheatingGroupMembers.objects.create(
            longitudinal_cheating_group=long_group,
            student=self.student,
            is_core_member=False,
            appearance_count=5,
        )
        LongitudinalCheatingGroupMembers.objects.create(
            longitudinal_cheating_group=long_group,
            student=self.student,
            is_core_member=True,
            appearance_count=2,
        )
        url = reverse("longitudinal-cheating-group-members-list")
        response = self.client.get(url, {"ordering": "appearance_count"})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        counts = [item["appearance_count"] for item in response.data["results"]]
        self.assertEqual(counts, sorted(counts))
