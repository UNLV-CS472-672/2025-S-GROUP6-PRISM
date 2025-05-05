"""
This module defines a Django management command to populate dummy data.

for CS472 (Capstone Project) with 3 sections, 6 cheaters per section,
and symmetric similarity scores among submissions.
"""

import random
from datetime import date, timedelta

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.db import transaction

from courses.models import (
    CourseCatalog,
    Semester,
    CourseInstances,
    Students,
    Professors,
)
from assignments.models import Assignments, Submissions
from cheating.models import SubmissionSimilarityPairs


class Command(BaseCommand):
    """
    Populate dummy data for CS472: 3 sections, 6 cheaters per section.

    symmetric similarity scores.
    """

    help = (
        "Populate dummy data for CS472 (3 sections) with 6 cheaters per "
        "section; symmetric similarity scores"
    )

    def handle(self, *args, **options):
        """
        Entry point for the management command.

        1) Ensure the Spring 2025 – Regular semester exists.
        2) Create CS472 course if needed.
        3) For each of 3 sections:
           a) Create a professor and course instance.
           b) Create 35 students (unique ACE/NSHE/CodeGrade IDs).
           c) Create 10 assignments spaced weekly starting Feb 1, 2025.
           d) Create one submission per student per assignment.
           e) Randomly label 6 students as “Cheater” per section.
           f) Seed symmetric similarity pairs for all submission pairs,
              with higher percentages among cheaters on chosen assignments.
        """
        with transaction.atomic():
            # 1) Semester setup
            semester, _ = Semester.objects.get_or_create(
                year=2025,
                term="Spring",
                session="Regular",
                defaults={"name": "Spring 2025 – Regular"},
            )

            User = get_user_model()
            base_nshe = 800_000
            base_codegd = 900_000

            def create_prof(section):
                """
                Create or fetch a dummy professor user for a section.

                Username is unique per section.
                """
                username = f"prof_cs472_{section}"
                email = f"{username}@example.edu"
                user, _ = User.objects.get_or_create(
                    username=username,
                    defaults={"email": email, "password": "pbkdf2_sha256$dummy"},
                )
                prof, _ = Professors.objects.get_or_create(user=user)
                return prof

            def get_course():
                """
                Fetch or create the CS472 CourseCatalog entry.

                Title set to 'Capstone Project'.
                """
                course, _ = CourseCatalog.objects.get_or_create(
                    subject="CS",
                    catalog_number=472,
                    defaults={
                        "name": "CS472",
                        "course_title": "Capstone Project",
                        "course_level": "400",
                    },
                )
                return course

            def create_students(start_index, count, prefix):
                """
                Create or fetch `count` Students, starting at `start_index`.

                ACE IDs use prefix + zero-padded index (e.g. CS472001).
                """
                students = []
                for i in range(count):
                    idx = start_index + i
                    ace_id = f"{prefix.upper()}{idx:03}"
                    email = f"{prefix.lower()}{idx}@example.edu"
                    s, _ = Students.objects.get_or_create(
                        ace_id=ace_id,
                        defaults={
                            "email": email,
                            "nshe_id": base_nshe + idx,
                            "codegrade_id": base_codegd + idx,
                            "first_name": "Student",
                            "last_name": str(idx),
                        },
                    )
                    students.append(s)
                return students

            def create_assignments(course, sem):
                """
                Create 10 weekly assignments for CS472 starting Feb 1, 2025.

                Returns the list of Assignments.
                """
                assigns = []
                start_due = date(2025, 2, 1)
                for num in range(1, 11):
                    due = start_due + timedelta(weeks=num - 1)
                    lock = due + timedelta(hours=2)
                    a, _ = Assignments.objects.get_or_create(
                        course_catalog=course,
                        semester=sem,
                        assignment_number=num,
                        defaults={
                            "title": f"Assignment {num}",
                            "due_date": due,
                            "lock_date": lock,
                            "has_base_code": False,
                            "moss_report_directory_path": f"/tmp/moss/CS472_{num}",
                            "bulk_ai_directory_path": f"/tmp/ai/CS472_{num}",
                            "language": "python",
                            "has_policy": False,
                            "pdf_filepath": f"/tmp/pdfs/CS472_{num}.pdf",
                        },
                    )
                    assigns.append(a)
                return assigns

            def add_submissions(assignments, students, inst):
                """
                Create one Submission per (assignment, student, instance).

                Returns a dict mapping (assignment_id, student_id) → Submission.
                """
                submap = {}
                for a in assignments:
                    for s in students:
                        sub, _ = Submissions.objects.get_or_create(
                            assignment=a,
                            student=s,
                            course_instance=inst,
                            defaults={
                                "grade": round(random.uniform(60, 100), 2),
                                "created_at": a.due_date - timedelta(days=1),
                                "flagged": False,
                                "file_path": (
                                    f"/submissions/{s.ace_id}/"
                                    f"a{a.assignment_number}.py"
                                ),
                            },
                        )
                        submap[(a.pk, s.pk)] = sub
                return submap

            def seed_symmetric_pairs(assignments, students, submap, num_cheaters):
                """
                Seed SubmissionSimilarityPairs for every unique pair of submissions.

                - Mark `num_cheaters` as Cheater per section.
                - On 4 random assignments, cheater↔cheater get high % (40–55).
                - All other pairs get 5–35%.
                - Ensures symmetry: (smaller_pk, larger_pk) only once.
                """
                # Pick cheaters and rename them
                cheaters = random.sample(students, num_cheaters)
                for c in cheaters:
                    c.first_name = "Cheater"
                    c.save(update_fields=["first_name"])

                # Pick 4 assignments where cheaters are strongly similar
                cheat_assigns = set(random.sample(assignments, 4))

                for a in assignments:
                    seen = set()
                    for i, s1 in enumerate(students):
                        for s2 in students[i + 1 :]:
                            sub1 = submap[(a.pk, s1.pk)]
                            sub2 = submap[(a.pk, s2.pk)]
                            # Order by PK to avoid duplicates
                            if sub1.pk > sub2.pk:
                                sub1, sub2 = sub2, sub1

                            key = (sub1.pk, sub2.pk)
                            if key in seen:
                                continue
                            seen.add(key)

                            # Determine similarity percentage and file name
                            if a in cheat_assigns and s1 in cheaters and s2 in cheaters:
                                pct = random.randint(40, 55)
                                fname = "__all__"
                            else:
                                pct = random.randint(5, 35)
                                fname = "sample.py"

                            SubmissionSimilarityPairs.objects.get_or_create(
                                assignment=a,
                                submission_id_1=sub1,
                                submission_id_2=sub2,
                                defaults={
                                    "file_name": fname,
                                    "match_id": random.randint(10000, 99999),
                                    "percentage": pct,
                                },
                            )

            # Main orchestration for CS472
            course = get_course()
            for section in (1, 2, 3):
                prof = create_prof(section)
                inst, _ = CourseInstances.objects.get_or_create(
                    course_catalog=course,
                    semester=semester,
                    section_number=section,
                    defaults={
                        "professor": prof,
                        "canvas_course_id": 7000 + section,
                    },
                )

                # Stepwise creation
                students = create_students(section * 100, 35, prefix="CS472")
                assignments = create_assignments(course, semester)
                submap = add_submissions(assignments, students, inst)
                seed_symmetric_pairs(assignments, students, submap, num_cheaters=6)

            self.stdout.write(
                self.style.SUCCESS(
                    "✅ Dummy CS472 data loaded with 3 sections — "
                    "Cheaters: 40–55%, Non-Cheaters: 5–35%, Symmetric!"
                )
            )
