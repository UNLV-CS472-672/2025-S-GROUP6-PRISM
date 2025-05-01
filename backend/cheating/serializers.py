"""Serializers for the Cheating app models."""

from rest_framework import serializers

from .models import (
    CheatingGroups,
    CheatingGroupMembers,
    ConfirmedCheaters,
    FlaggedStudents,
    SubmissionSimilarityPairs,
    LongitudinalCheatingGroups,
    LongitudinalCheatingGroupMembers,
    LongitudinalCheatingGroupInstances,
    StudentReport,
    AssignmentReport,
)


class CheatingGroupsSerializer(serializers.ModelSerializer):
    """Serializer for the CheatingGroups model."""

    class Meta:
        """Meta options for CheatingGroupsSerializer.

        Attributes:
            model: The CheatingGroups model.
            fields: All model fields.
        """

        model = CheatingGroups
        fields = "__all__"


class CheatingGroupMembersSerializer(serializers.ModelSerializer):
    """Serializer for the CheatingGroupMembers model."""

    class Meta:
        """Meta options for CheatingGroupMembersSerializer.

        Attributes:
            model: The CheatingGroupMembers model.
            fields: All model fields.
        """

        model = CheatingGroupMembers
        fields = "__all__"


class ConfirmedCheatersSerializer(serializers.ModelSerializer):
    """Serializer for the ConfirmedCheaters model."""

    class Meta:
        """Meta options for ConfirmedCheatersSerializer.

        Attributes:
            model: The ConfirmedCheaters model.
            fields: All model fields.
        """

        model = ConfirmedCheaters
        fields = "__all__"


class FlaggedStudentsSerializer(serializers.ModelSerializer):
    """Serializer for the FlaggedStudents model."""

    class Meta:
        """Meta options for FlaggedStudentsSerializer.

        Attributes:
            model: The FlaggedStudents model.
            fields: All model fields.
        """

        model = FlaggedStudents
        fields = "__all__"


class SubmissionSimilarityPairsSerializer(serializers.ModelSerializer):
    """Serializer for the SubmissionSimilarityPairs model."""

    class Meta:
        """Meta options for SubmissionSimilarityPairsSerializer.

        Attributes:
            model: The SubmissionSimilarityPairs model.
            fields: All model fields.
        """

        model = SubmissionSimilarityPairs
        fields = "__all__"


class LongitudinalCheatingGroupsSerializer(serializers.ModelSerializer):
    """Serializer for the LongitudinalCheatingGroups model."""

    class Meta:
        """Meta options for LongitudinalCheatingGroupsSerializer.

        Attributes:
            model: The LongitudinalCheatingGroups model.
            fields: All model fields.
        """

        model = LongitudinalCheatingGroups
        fields = "__all__"


class LongitudinalCheatingGroupMembersSerializer(serializers.ModelSerializer):
    """Serializer for the LongitudinalCheatingGroupMembers model."""

    class Meta:
        """Meta options for LongitudinalCheatingGroupMembersSerializer.

        Attributes:
            model: The LongitudinalCheatingGroupMembers model.
            fields: All model fields.
        """

        model = LongitudinalCheatingGroupMembers
        fields = "__all__"


class LongitudinalCheatingGroupInstancesSerializer(serializers.ModelSerializer):
    """Serializer for the LongitudinalCheatingGroupInstances model."""

    class Meta:
        """Meta options for LongitudinalCheatingGroupInstancesSerializer.

        Attributes:
            model: The LongitudinalCheatingGroupInstances model.
            fields: All model fields.
        """

        model = LongitudinalCheatingGroupInstances
        fields = "__all__"


class StudentReportSerializer(serializers.ModelSerializer):
    """Serializes one StudentReport instance."""

    submission_id = serializers.IntegerField(
        source="submission_id", read_only=True, help_text="PK of the student submission"
    )
    student_name = serializers.SerializerMethodField(
        help_text="Full name of the student"
    )

    class Meta:
        """Meta options for StudentReportSerializer."""

        model = StudentReport
        fields = (
            "submission_id",
            "student_name",
            "mean_similarity",
            "z_score",
            "ci_lower",
            "ci_upper",
        )

    def get_student_name(self, obj):
        """Look up and return student’s full name."""
        student = obj.submission.student
        return f"{student.first_name} {student.last_name}"


class AssignmentReportSerializer(serializers.ModelSerializer):
    """Serializes an AssignmentReport along with its StudentReports."""

    assignment_id = serializers.IntegerField(
        source="assignment_id", read_only=True, help_text="PK of the assignment"
    )
    student_reports = StudentReportSerializer(
        many=True, read_only=True, help_text="List of per-student inference results"
    )

    class Meta:
        """Meta options for AssignmentReportSerializer."""

        model = AssignmentReport
        fields = (
            "id",
            "assignment_id",
            "created_at",
            "mu",
            "sigma",
            "variance",
            "student_reports",
        )
        read_only_fields = (
            "id",
            "assignment_id",
            "created_at",
            "student_reports",
        )
