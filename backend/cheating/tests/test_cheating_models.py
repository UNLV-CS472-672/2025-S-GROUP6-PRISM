"""Unit tests for cheating detection models."""

import datetime

import pytest
from django.apps import apps
from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from cheating.models import (
    CheatingGroups,
    CheatingGroupMembers,
    ConfirmedCheaters,
    FlaggedStudents,
    SubmissionSimilarityPairs,
    LongitudinalCheatingGroups,
    LongitudinalCheatingGroupMembers,
    LongitudinalCheatingGroupInstances,
)

User = get_user_model()
Semester = apps.get_model("courses", "Semester")
CourseCatalog = apps.get_model("courses", "CourseCatalog")
Professors = apps.get_model("courses", "Professors")
TeachingAssistants = apps.get_model("courses", "TeachingAssistants")
CourseInstances = apps.get_model("courses", "CourseInstances")
Students = apps.get_model("courses", "Students")
Assignments = apps.get_model("assignments", "Assignments")
Submissions = apps.get_model("assignments", "Submissions")


class CheatingModelsTest(TestCase):
    """Full coverage for every cheating detection model."""

    def setUp(self):
        """Create a course instance, students, and an assignment."""
        self.sem = Semester.objects.create(
            name="F23", year=2023, term="Fall", session="Regular"
        )
        self.cat = CourseCatalog.objects.create(
            name="CS200",
            subject="CS",
            catalog_number=200,
            course_title="Algo",
            course_level="Undergrad",
        )
        prof_user = User.objects.create_user(
            username="prof", password="p", email="p@x.com"
        )
        self.prof = Professors.objects.create(user=prof_user)
        ta_user = User.objects.create_user(
            username="ta", password="p", email="ta@x.com"
        )
        self.ta = TeachingAssistants.objects.create(user=ta_user)
        self.cinst = CourseInstances.objects.create(
            semester=self.sem,
            course_catalog=self.cat,
            section_number=1,
            professor=self.prof,
            teaching_assistant=self.ta,
            canvas_course_id=42,
        )
        self.st1 = Students.objects.create(
            email="a@x.com",
            nshe_id=1,
            codegrade_id=11,
            ace_id="A1",
            first_name="A",
            last_name="One",
        )
        self.st2 = Students.objects.create(
            email="b@x.com",
            nshe_id=2,
            codegrade_id=22,
            ace_id="B2",
            first_name="B",
            last_name="Two",
        )
        self.asg = Assignments.objects.create(
            course_catalog=self.cat,
            semester=self.sem,
            assignment_number=1,
            title="Test",
            due_date=datetime.date.today(),
            pdf_filepath="p.pdf",
            has_base_code=False,
            moss_report_directory_path="m/",
            bulk_ai_directory_path="b/",
            language="Py",
            has_policy=False,
        )
        self.sub1 = Submissions.objects.create(
            grade=1,
            created_at=datetime.date.today(),
            flagged=False,
            assignment=self.asg,
            student=self.st1,
            course_instance=self.cinst,
            file_path="s1.py",
        )
        self.sub2 = Submissions.objects.create(
            grade=2,
            created_at=datetime.date.today(),
            flagged=False,
            assignment=self.asg,
            student=self.st2,
            course_instance=self.cinst,
            file_path="s2.py",
        )

    def test_cheating_groups_str_and_unique(self):
        """Test __str__ for CheatingGroups."""
        cg = CheatingGroups.objects.create(
            assignment=self.asg,
            cohesion_score=0.1234,
            analysis_report_path="r1.html",
        )
        assert "Cohesion Score: 0.12" in str(cg)
        with pytest.raises(IntegrityError):
            CheatingGroups.objects.create(
                assignment=self.asg,
                cohesion_score=0.5,
                analysis_report_path="r1.html",
            )

    def test_cheating_group_members_str(self):
        """Test __str__ for CheatingGroupMembers."""
        cg = CheatingGroups.objects.create(
            assignment=self.asg, cohesion_score=0.5, analysis_report_path="r2.html"
        )
        cgm = CheatingGroupMembers.objects.create(
            cheating_group=cg, student=self.st1, cluster_distance=5.6789
        )
        assert "Distance: 5.68" in str(cgm)

    def test_confirmed_cheaters_str_and_unique(self):
        """Test __str__ for ConfirmedCheaters."""
        cc = ConfirmedCheaters.objects.create(
            confirmed_date=datetime.date(2023, 1, 1),
            threshold_used=90,
            assignment=self.asg,
            student=self.st1,
        )
        assert "(Confirmed: 2023-01-01)" in str(cc)
        with pytest.raises(IntegrityError):
            ConfirmedCheaters.objects.create(
                confirmed_date=datetime.date(2023, 1, 2),
                threshold_used=80,
                assignment=self.asg,
                student=self.st1,
            )

    def test_submission_similarity_pairs_str_and_unique(self):
        """Test __str__ for SubmissionSimilarityPairs."""
        ssp = SubmissionSimilarityPairs.objects.create(
            assignment=self.asg,
            file_name="f.py",
            submission_id_1=self.sub1,
            submission_id_2=self.sub2,
            match_id=99,
            percentage=88,
        )
        out = str(ssp)
        assert "↔" in out and "(88%)" in out
        with pytest.raises(IntegrityError):
            SubmissionSimilarityPairs.objects.create(
                assignment=self.asg,
                file_name="g.py",
                submission_id_1=self.sub1,
                submission_id_2=self.sub2,
                match_id=100,
                percentage=77,
            )

    def test_flagged_students_str_and_unique(self):
        """Test __str__ for FlaggedStudents."""
        ssp = SubmissionSimilarityPairs.objects.create(
            assignment=self.asg,
            file_name="f2.py",
            submission_id_1=self.sub1,
            submission_id_2=self.sub2,
            match_id=101,
            percentage=50,
        )
        fs = FlaggedStudents.objects.create(
            professor=self.prof,
            student=self.st2,
            similarity=ssp,
            generative_ai=True,
        )
        assert "(AI)" in str(fs)
        with pytest.raises(IntegrityError):
            FlaggedStudents.objects.create(
                professor=self.prof,
                student=self.st2,
                similarity=ssp,
                generative_ai=False,
            )

    def test_longitudinal_models_strisms_and_uniques(self):
        """Test __str__ for LongitudinalCheatingGroups and members."""
        lg = LongitudinalCheatingGroups.objects.create(score=3.1415)
        assert "Score: 3.14" in str(lg)

        lgm = LongitudinalCheatingGroupMembers.objects.create(
            longitudinal_cheating_group=lg,
            student=self.st1,
            is_core_member=False,
            appearance_count=7,
        )
        assert "Peripheral" in str(lgm)

        cg = CheatingGroups.objects.create(
            assignment=self.asg, cohesion_score=0.2, analysis_report_path="r3.html"
        )
        lci = LongitudinalCheatingGroupInstances.objects.create(
            cheating_group=cg,
            longitudinal_cheating_group=lg,
        )
        assert "LongitudinalGroup" in str(lci)

    def test_full_cycle_combined(self):
        """Touch every model in one transaction to ensure nothing blows up under real‐world ordering."""
        cg = CheatingGroups(
            assignment=self.asg, cohesion_score=0.999, analysis_report_path="all.html"
        )
        cg.save()
        cgm = CheatingGroupMembers.objects.create(
            cheating_group=cg, student=self.st1, cluster_distance=0.001
        )
        cc = ConfirmedCheaters.objects.create(
            confirmed_date=datetime.date.today(),
            threshold_used=1,
            assignment=self.asg,
            student=self.st2,
        )
        ssp = SubmissionSimilarityPairs.objects.create(
            assignment=self.asg,
            file_name="z.py",
            submission_id_1=self.sub1,
            submission_id_2=self.sub2,
            match_id=111,
            percentage=99,
        )
        fs = FlaggedStudents.objects.create(
            professor=self.prof,
            student=self.st1,
            similarity=ssp,
            generative_ai=False,
        )
        lg = LongitudinalCheatingGroups.objects.create(score=2.718)
        lci = LongitudinalCheatingGroupInstances.objects.create(
            cheating_group=cg, longitudinal_cheating_group=lg
        )
        lgm = LongitudinalCheatingGroupMembers.objects.create(
            longitudinal_cheating_group=lg,
            student=self.st2,
            is_core_member=True,
            appearance_count=3,
        )
        for obj in (cg, cgm, cc, ssp, fs, lg, lci, lgm):
            assert str(obj)
