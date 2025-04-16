# Generated by Django 5.1.5 on 2025-04-16 20:43

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("assignments", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Professors",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Semester",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.TextField()),
                ("year", models.SmallIntegerField()),
                ("term", models.TextField()),
                ("session", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="StudentEnrollments",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Students",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("email", models.TextField(unique=True)),
                ("nshe_id", models.BigIntegerField(unique=True)),
                (
                    "codegrade_id",
                    models.BigIntegerField(db_column="codeGrade_id", unique=True),
                ),
                ("ace_id", models.TextField(unique=True)),
                ("first_name", models.TextField()),
                ("last_name", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="TeachingAssistantEnrollment",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="TeachingAssistants",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="CourseCatalog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=50, unique=True)),
                ("subject", models.TextField()),
                ("catalog_number", models.SmallIntegerField()),
                ("course_title", models.CharField(max_length=255)),
                ("course_level", models.TextField()),
            ],
            options={
                "unique_together": {("subject", "catalog_number")},
            },
        ),
        migrations.CreateModel(
            name="CourseInstances",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("section_number", models.IntegerField()),
                ("canvas_course_id", models.BigIntegerField(unique=True)),
                (
                    "course_catalog",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="courses.coursecatalog",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="CourseAssignmentCollaboration",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "assignment",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="assignments.assignments",
                    ),
                ),
                (
                    "course_instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="courses.courseinstances",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="ProfessorEnrollments",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "course_instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="courses.courseinstances",
                    ),
                ),
            ],
        ),
    ]
