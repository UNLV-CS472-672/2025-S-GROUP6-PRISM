"""
This module defines a Django management command to populate dummy data.

for CS481 (Capstone in Cybersecurity) with 5 sections, a total of 4 cheaters,
and symmetric similarity scores across submissions.
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
    Populate dummy data for CS481.

    - 5 sections
    - 4 cheaters total
    - symmetric similarity scores among submissions
    """

    help = (
        "Populate dummy data for CS481 with 5 sections, 4 cheaters total, "
        "symmetric similarities"
    )

    def handle(self, *args, **options):
        """
        1) Ensure Spring 2025 – Regular semester exists.

        2) Create or fetch CS481 course entry.
        3) Loop over 5 sections:
           a) Create a dummy professor and course instance.
           b) Create 30 students per section with unique IDs.
           c) Create 8 weekly assignments starting Jan 20, 2025.
           d) Generate one submission per student per assignment.
           e) Randomly assign up to 4 total cheaters (across sections),
              relabel them, and seed high-similarity pairs on 2 assignments.
           f) Seed all other submission pairs with lower similarities.
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
            base_nshe = 700_000
            base_codegd = 800_000

            def create_prof(section):
                """Create or fetch a dummy professor for the given section."""
                username = f"prof_cs481_{section}"
                email = f"{username}@example.edu"
                user, _ = User.objects.get_or_create(
                    username=username,
                    defaults={"email": email, "password": "pbkdf2_sha256$dummy"},
                )
                prof, _ = Professors.objects.get_or_create(user=user)
                return prof

            def get_course():
                """Fetch or create the CS481 catalog entry."""
                course, _ = CourseCatalog.objects.get_or_create(
                    subject="CS",
                    catalog_number=481,
                    defaults={
                        "name": "CS481",
                        "course_title": "Capstone in Cybersecurity",
                        "course_level": "400",
                    },
                )
                return course

            def create_students(start_index, count):
                """Create or fetch `count` students starting at given index."""
                students = []
                for i in range(count):
                    idx = start_index + i
                    ace_id = f"CS481{idx:03}"
                    email = f"cs481_{idx}@example.edu"
                    student, _ = Students.objects.get_or_create(
                        ace_id=ace_id,
                        defaults={
                            "email": email,
                            "nshe_id": base_nshe + idx,
                            "codegrade_id": base_codegd + idx,
                            "first_name": "Student",
                            "last_name": str(idx),
                        },
                    )
                    students.append(student)
                return students

            def create_assignments(course, sem):
                """Create 8 weekly assignments starting Jan 20, 2025."""
                assignments = []
                start_due = date(2025, 1, 20)
                for num in range(1, 9):
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
                            "language": "python",
                            "pdf_filepath": f"/tmp/pdfs/CS481_{num}.pdf",
                            "moss_report_directory_path": f"/tmp/moss/CS481_{num}",
                            "bulk_ai_directory_path": f"/tmp/ai/CS481_{num}",
                            "has_policy": False,
                        },
                    )
                    assignments.append(a)
                return assignments

            def add_submissions(assignments, students, instance):
                """
                Create one submission per student per assignment.

                Returns a mapping for seeding pairs.
                """
                submap = {}
                for a in assignments:
                    for s in students:
                        sub, _ = Submissions.objects.get_or_create(
                            assignment=a,
                            student=s,
                            course_instance=instance,
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

            def create_symmetrical_pairs(assignments, students, submap, cheaters):
                """
                Seed SubmissionSimilarityPairs symmetrically.

                - Cheaters get 40–55% on 2 random assignments.
                - All other pairs get 5–35%.
                """
                # Select 2 assignments for high-sim among cheaters
                cheat_assigns = set(random.sample(assignments, 2))

                for a in assignments:
                    # High similarity for cheater↔cheater on chosen assignments
                    if a in cheat_assigns:
                        for i in range(len(cheaters)):
                            for j in range(i + 1, len(cheaters)):
                                s1 = submap[(a.pk, cheaters[i].pk)]
                                s2 = submap[(a.pk, cheaters[j].pk)]
                                if s1.pk > s2.pk:
                                    s1, s2 = s2, s1
                                SubmissionSimilarityPairs.objects.get_or_create(
                                    assignment=a,
                                    submission_id_1=s1,
                                    submission_id_2=s2,
                                    defaults={
                                        "file_name": "__all__",
                                        "match_id": random.randint(10_000, 99_999),
                                        "percentage": random.randint(40, 55),
                                    },
                                )

                    # Lower similarity for all other pairs
                    for i, sA in enumerate(students):
                        for sB in students[i + 1 :]:
                            if a in cheat_assigns and sA in cheaters and sB in cheaters:
                                # Already created above
                                continue
                            sub1 = submap[(a.pk, sA.pk)]
                            sub2 = submap[(a.pk, sB.pk)]
                            if sub1.pk > sub2.pk:
                                sub1, sub2 = sub2, sub1
                            SubmissionSimilarityPairs.objects.get_or_create(
                                assignment=a,
                                submission_id_1=sub1,
                                submission_id_2=sub2,
                                defaults={
                                    "file_name": "sample.py",
                                    "match_id": random.randint(20_000, 29_999),
                                    "percentage": random.randint(5, 35),
                                },
                            )

            # ───────────────── EXECUTION START ─────────────────
            course = get_course()
            total_cheaters = 4
            cheaters_remaining = total_cheaters
            idx_offset = 0

            for section in range(1, 6):
                prof = create_prof(section)
                instance, _ = CourseInstances.objects.get_or_create(
                    course_catalog=course,
                    semester=semester,
                    section_number=section,
                    defaults={
                        "professor": prof,
                        "canvas_course_id": 9000 + section,
                    },
                )

                # Create students and assignments
                students = create_students(idx_offset * 100, 30)
                idx_offset += 1
                assignments = create_assignments(course, semester)
                submap = add_submissions(assignments, students, instance)

                # Pick up to remaining cheaters from this section
                cheaters_here = []
                if cheaters_remaining > 0:
                    pick = min(cheaters_remaining, len(students))
                    cheaters_here = random.sample(students, pick)
                    for c in cheaters_here:
                        c.first_name = "Cheater"
                        c.save(update_fields=["first_name"])
                    cheaters_remaining -= pick

                create_symmetrical_pairs(assignments, students, submap, cheaters_here)

            self.stdout.write(
                self.style.SUCCESS(
                    "✅ Dummy CS481 data created for 5 sections — "
                    "4 cheaters total, symmetric sim scores"
                )
            )
