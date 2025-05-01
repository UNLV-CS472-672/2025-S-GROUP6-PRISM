"""Serializers for the Assignments app."""

from rest_framework import serializers

from .models import (
    Assignments,
    Submissions,
    BaseFiles,
    BulkSubmissions,
    Constraints,
    PolicyViolations,
    RequiredSubmissionFiles,
)


class AssignmentsSerializer(serializers.ModelSerializer):
    """Serializer for the Assignments model."""

    class Meta:
        """Meta options for the AssignmentsSerializer.

        Attributes:
            model: The model class that this serializer serializes.
            fields: All model fields are included.
        """

        model = Assignments
        fields = "__all__"


class SubmissionsSerializer(serializers.ModelSerializer):
    """Serializer for the Submissions model."""

    class Meta:
        """Meta options for the SubmissionsSerializer.

        Attributes:
            model: The model class that this serializer serializes.
            fields: All model fields are included.
        """

        model = Submissions
        fields = "__all__"


class BaseFilesSerializer(serializers.ModelSerializer):
    """Serializer for the BaseFiles model."""

    class Meta:
        """Meta options for the BaseFilesSerializer.

        Attributes:
            model: The model class that this serializer serializes.
            fields: All model fields are included.
        """

        model = BaseFiles
        fields = "__all__"


class BulkSubmissionsSerializer(serializers.ModelSerializer):
    """Serializer for the BulkSubmissions model."""

    class Meta:
        """Meta options for the BulkSubmissionsSerializer.

        Attributes:
            model: The model class that this serializer serializes.
            fields: All model fields are included.
        """

        model = BulkSubmissions
        fields = "__all__"


class ConstraintsSerializer(serializers.ModelSerializer):
    """Serializer for the Constraints model."""

    class Meta:
        """Meta options for the ConstraintsSerializer.

        Attributes:
            model: The model class that this serializer serializes.
            fields: All model fields are included.
        """

        model = Constraints
        fields = "__all__"


class PolicyViolationsSerializer(serializers.ModelSerializer):
    """Serializer for the PolicyViolations model."""

    class Meta:
        """Meta options for the PolicyViolationsSerializer.

        Attributes:
            model: The model class that this serializer serializes.
            fields: All model fields are included.
        """

        model = PolicyViolations
        fields = "__all__"


class RequiredSubmissionFilesSerializer(serializers.ModelSerializer):
    """Serializer for the RequiredSubmissionFiles model."""

    class Meta:
        """Meta options for the RequiredSubmissionFilesSerializer.

        Attributes:
            model: The model class that this serializer serializes.
            fields: All model fields are included.
        """

        model = RequiredSubmissionFiles
        fields = "__all__"
