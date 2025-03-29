"""Serializers for Assignment-related models.

This module defines Django REST Framework serializers for managing students,
assignments, submissions, flagged submissions, confirmed cheaters, and
flagged students. It supports nested representation for read operations and
primary key relationships for write operations.
"""

from rest_framework import serializers
from assignments import models
from courses import models as courses_models
from courses import serializers as courses_serializer


class StudentSerializer(serializers.ModelSerializer):
    """Serialize Student model instances."""

    class Meta:
        """Meta options for StudentSerializer."""

        model = models.Student
        fields = "__all__"


class FlaggedStudentSerializer(serializers.ModelSerializer):
    """Serialize and deserialize FlaggedStudent model instances."""

    student = StudentSerializer(read_only=True)
    student_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Student.objects.all(),
        source="student",
        write_only=True,
    )

    class Meta:
        """Meta options for FlaggedStudentSerializer."""

        model = models.FlaggedStudent
        fields = "__all__"


class AssignmentSerializer(serializers.ModelSerializer):
    """Serialize and deserialize Assignment model instances."""

    professor = courses_serializer.ProfessorSerializer(read_only=True)
    class_instance = courses_serializer.ClassSerializer(read_only=True)

    professor_id = serializers.PrimaryKeyRelatedField(
        queryset=courses_models.Professor.objects.all(),
        source="professor",
        write_only=True,
    )
    class_instance_id = serializers.PrimaryKeyRelatedField(
        queryset=courses_models.Class.objects.all(),
        source="class_instance",
        write_only=True,
    )

    class Meta:
        """Meta options for AssignmentSerializer."""

        model = models.Assignment
        fields = "__all__"


class SubmissionSerializer(serializers.ModelSerializer):
    """Serialize and deserialize Submission model instances."""

    professor = courses_serializer.ProfessorSerializer(read_only=True)
    student = StudentSerializer(read_only=True)
    assignment = AssignmentSerializer(read_only=True)

    professor_id = serializers.PrimaryKeyRelatedField(
        queryset=courses_models.Professor.objects.all(),
        source="professor",
        write_only=True,
    )
    assignment_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Assignment.objects.all(),
        source="assignment",
        write_only=True,
    )
    student_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Student.objects.all(),
        source="student",
        write_only=True,
    )

    class Meta:
        """Meta options for SubmissionSerializer."""

        model = models.Submission
        fields = "__all__"


class FlaggedSubmissionSerializer(serializers.ModelSerializer):
    """Serialize and deserialize FlaggedSubmission model instances."""

    similarity_with = StudentSerializer(read_only=True, many=True)
    submission = SubmissionSerializer(read_only=True)

    student_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Student.objects.all(),
        source="student",
        write_only=True,
    )
    submission_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Submission.objects.all(),
        source="submission",
        write_only=True,
    )

    class Meta:
        """Meta options for FlaggedSubmissionSerializer."""

        model = models.FlaggedSubmission
        fields = "__all__"


class ConfirmedCheaterSerializer(serializers.ModelSerializer):
    """Serialize and deserialize ConfirmedCheater model instances."""

    student = StudentSerializer(read_only=True)
    professor = courses_serializer.ProfessorSerializer(read_only=True)

    student_id = serializers.PrimaryKeyRelatedField(
        queryset=models.Student.objects.all(),
        source="student",
        write_only=True,
    )
    professor_id = serializers.PrimaryKeyRelatedField(
        queryset=courses_models.Professor.objects.all(),
        source="professor",
        write_only=True,
    )

    class Meta:
        """Meta options for ConfirmedCheaterSerializer."""

        model = models.ConfirmedCheater
        fields = "__all__"
