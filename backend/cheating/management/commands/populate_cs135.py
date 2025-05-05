"""Command to populate a fresh database with dummy data for CS135 course."""

import random
from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
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

User = get_user_model()


class Command(BaseCommand):
    """Command to populate a fresh database with dummy data for CS135 course."""

    help = (
        "On a fresh DB, populate CS135 with:\n"
        "- 2 sections × 40 students each\n"
        "- 8 cheaters (first_name='Cheater')\n"
        "- 10 dummy assignments\n"
        "- 800 submissions (unique file_path per submission)\n"
        "- 31,600 unique similarity pairs\n"
        "Logs ⚠️ if any lookup ever fails."
    )

    def handle(self, *args, **opts):
        """Entry point for the command."""
        with transaction.atomic():
            # 1) Semester & Catalog
            semester, _ = Semester.objects.get_or_create(
                year=2025,
                term="Spring",
                session="Regular",
                defaults={"name": "Spring 2025 – Regular"},
            )
            course, _ = CourseCatalog.objects.get_or_create(
                subject="CS",
                catalog_number=135,
                defaults={
                    "name": "CS135",
                    "course_title": "Dummy CS135",
                    "course_level": "100",
                },
            )
            self.stdout.write("📖 Ensured semester & course catalog")

            # 2) Two sections + 40 students each
            all_students = []
            for sec in (1, 2):
                uname = f"prof_cs135_s{sec}"
                user, created = User.objects.get_or_create(
                    username=uname,
                    defaults={"email": f"{uname}@example.edu"},
                )
                if created:
                    user.set_unusable_password()
                    user.save(update_fields=["password"])

                prof, _ = Professors.objects.get_or_create(user=user)
                inst, _ = CourseInstances.objects.get_or_create(
                    course_catalog=course,
                    semester=semester,
                    section_number=sec,
                    defaults={"professor": prof, "canvas_course_id": 135000 + sec},
                )

                base = (sec - 1) * 40
                for i in range(40):
                    ace = f"{sec}{i + 1:02d}"
                    stud, _ = Students.objects.get_or_create(
                        ace_id=ace,
                        defaults={
                            "email": f"{ace}@example.edu",
                            "nshe_id": 900000 + base + i,
                            "codegrade_id": 1000000 + base + i,
                            "first_name": "Student",
                            "last_name": str(i + 1),
                        },
                    )
                    all_students.append((stud, inst))

            self.stdout.write(f"👥 Ensured {len(all_students)} students")

            # 3) Pick & label 8 cheaters
            studs = [s for s, _ in all_students]
            cheaters = set(random.sample(studs, 8))
            for c in cheaters:
                c.first_name = "Cheater"
                c.last_name = c.ace_id
                c.save(update_fields=["first_name", "last_name"])
            self.stdout.write(f"🚩 Marked {len(cheaters)} cheaters")

            sorted_cheat = sorted(cheaters, key=lambda s: s.pk)
            cheat_pairs = [
                (sorted_cheat[i], sorted_cheat[j])
                for i in range(len(sorted_cheat))
                for j in range(i + 1, len(sorted_cheat))
            ]

            # 4) Ten dummy assignments
            assignments = []
            start = date(2025, 1, 15)
            for num in range(1, 11):
                due = start + timedelta(weeks=num - 1)
                lock = due + timedelta(hours=2)
                a, created = Assignments.objects.get_or_create(
                    course_catalog=course,
                    semester=semester,
                    assignment_number=num,
                    defaults={
                        "title": f"Assignment {num}",
                        "due_date": due,
                        "lock_date": lock,
                        "has_base_code": False,
                        "language": "python",
                        "pdf_filepath": f"dummy/assignment_{num}.pdf",
                        "moss_report_directory_path": f"dummy/moss_reports/assignment_{num}",
                        "bulk_ai_directory_path": f"dummy/bulk_ai/assignment_{num}",
                        "has_policy": False,
                    },
                )
                assignments.append(a)
                if created:
                    self.stdout.write(f"📄 Created assignment #{num}")
            self.stdout.write(f"📚 Ensured {len(assignments)} assignments")

            # 5) Bulk‐create 800 submissions with unique file paths
            subs = []
            for stud, inst in all_students:
                for a in assignments:
                    # unique file_path per submission
                    path = f"dummy/{stud.ace_id}_a{a.assignment_number}.py"
                    subs.append(
                        Submissions(
                            assignment=a,
                            student=stud,
                            course_instance=inst,
                            grade=round(random.uniform(60, 100), 2),
                            created_at=a.due_date - timedelta(days=1),
                            flagged=False,
                            file_path=path,
                        )
                    )
            Submissions.objects.bulk_create(subs)
            self.stdout.write(f"✉️ Created {len(subs)} submissions")

            # 6) Build submission lookup by (assignment_number, ace_id)
            submap = {}
            for sub in Submissions.objects.filter(
                assignment__in=assignments
            ).select_related("assignment", "student"):
                key = (sub.assignment.assignment_number, sub.student.ace_id)
                submap[key] = sub.pk

            # 7) Bulk‐create similarity pairs, logging any lookup misses
            pairs = []
            n = len(studs)
            for a in assignments:
                num = a.assignment_number

                # cheater–cheater pairs
                for c1, c2 in cheat_pairs:
                    k1, k2 = (num, c1.ace_id), (num, c2.ace_id)
                    if k1 not in submap or k2 not in submap:
                        self.stderr.write(f"⚠️ Missing sub for cheaters {k1},{k2}")
                        continue
                    s1, s2 = submap[k1], submap[k2]
                    pairs.append(
                        SubmissionSimilarityPairs(
                            assignment_id=a.pk,
                            submission_id_1_id=min(s1, s2),
                            submission_id_2_id=max(s1, s2),
                            file_name="cheater_pair",
                            match_id=random.randint(10000, 19999),
                            percentage=random.randint(45, 75),
                        )
                    )

                # non-cheater pairs
                for i in range(n):
                    for j in range(i + 1, n):
                        s1_obj, s2_obj = studs[i], studs[j]
                        if s1_obj in cheaters and s2_obj in cheaters:
                            continue
                        k1, k2 = (num, s1_obj.ace_id), (num, s2_obj.ace_id)
                        if k1 not in submap or k2 not in submap:
                            self.stderr.write(f"⚠️ Missing sub for pair {k1},{k2}")
                            continue
                        s1, s2 = submap[k1], submap[k2]
                        pairs.append(
                            SubmissionSimilarityPairs(
                                assignment_id=a.pk,
                                submission_id_1_id=min(s1, s2),
                                submission_id_2_id=max(s1, s2),
                                file_name="sample.py",
                                match_id=random.randint(20000, 29999),
                                percentage=random.randint(30, 65),
                            )
                        )

            SubmissionSimilarityPairs.objects.bulk_create(pairs)
            self.stdout.write(f"🔗 Created {len(pairs)} similarity pairs")
            self.stdout.write(
                self.style.SUCCESS("✅ Dummy CS135 data fully populated!")
            )
