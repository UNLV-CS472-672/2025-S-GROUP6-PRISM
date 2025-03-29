"""Serializers for Courses models.

This module defines Django REST Framework serializers for the following models:
- Professor
- Class
- Semester
- ProfessorClassSection
- Enrollment

Each serializer is responsible for validating input and formatting output
according to frontend/API requirements. Nested representations are included
where appropriate, and PrimaryKeyRelatedFields are used for write operations.
"""

from rest_framework import serializers
from courses import models
from users import serializers as user_serializer


class ProfessorSerializer(serializers.ModelSerializer):
    """Serialize Professor model with nested User data."""

    user = user_serializer.UserSerializer(read_only=True)

    class Meta:
        """Meta options for ProfessorSerializer."""

        model = models.Professor
        fields = "__all__"


class ClassSerializer(serializers.ModelSerializer):
    """Serialize Class model."""

    class Meta:
        """Meta options for ClassSerializer."""

        model = models.Class
        fields = ["name"]


class SemesterSerializer(serializers.ModelSerializer):
    """Serialize Semester model."""

    class Meta:
        """Meta options for SemesterSerializer."""

        model = models.Semester
        fields = ["name"]


class ProfessorClassSectionSerializer(serializers.ModelSerializer):
    """Serialize ProfessorClassSection model.

    Includes both read-only nested serializers and write-only ID fields
    for related objects.
    """

    semester = SemesterSerializer(read_only=True)
    class_instance = ClassSerializer(read_only=True)

    semester_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Semester.objects.all(),
        source="semester",
        write_only=True,
    )
    class_instance_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Class.objects.all(),
        source="class_instance",
        write_only=True,
    )
    professor_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Professor.objects.all(),
        source="professor",
        write_only=True,
    )

    class Meta:
        """Meta options for ProfessorClassSectionSerializer."""

        model = models.ProfessorClassSection
        fields = [
            "section_number",
            "semester",
            "semester_id",
            "class_instance",
            "class_instance_id",
            "professor_id",
        ]


class EnrollmentSerializer(serializers.ModelSerializer):
    """Serialize Enrollment model for read/write operations.

    Shows related object names for readability and accepts IDs for write.
    """

    student = serializers.StringRelatedField(read_only=True)
    class_instance = serializers.StringRelatedField(read_only=True)
    semester = serializers.StringRelatedField(read_only=True)

    student_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Student.objects.all(),
        source="student",
        write_only=True,
    )
    class_instance_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Class.objects.all(),
        source="class_instance",
        write_only=True,
    )
    semester_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Semester.objects.all(),
        source="semester",
        write_only=True,
    )

    class Meta:
        """Meta options for EnrollmentSerializer."""

        model = models.Enrollment
        fields = [
            "id",
            "student",
            "student_id",
            "class_instance",
            "class_instance_id",
            "semester",
            "semester_id",
            "enrolled_date",
            "dropped",
            "dropped_date",
        ]
