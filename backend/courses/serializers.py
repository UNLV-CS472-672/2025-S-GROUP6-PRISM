"""Serializers for the Courses app."""

from rest_framework import serializers

from .models import (
    CourseCatalog,
    CourseInstances,
    Semester,
    CourseAssignmentCollaboration,
    Students,
    StudentEnrollments,
    Professors,
    ProfessorEnrollments,
    TeachingAssistants,
    TeachingAssistantEnrollment,
)


class CourseCatalogSerializer(serializers.ModelSerializer):
    """Serializer for the CourseCatalog model."""

    class Meta:
        """Meta options for CourseCatalogSerializer.

        Attributes:
            model: The CourseCatalog model.
            fields: All model fields.
        """

        model = CourseCatalog
        fields = "__all__"


class CourseInstancesSerializer(serializers.ModelSerializer):
    """Serializer for the CourseInstances model."""

    class Meta:
        """Meta options for CourseInstancesSerializer.

        Attributes:
            model: The CourseInstances model.
            fields: All model fields.
        """

        model = CourseInstances
        fields = "__all__"


class SemesterSerializer(serializers.ModelSerializer):
    """Serializer for the Semester model."""

    class Meta:
        """Meta options for SemesterSerializer.

        Attributes:
            model: The Semester model.
            fields: All model fields.
        """

        model = Semester
        fields = "__all__"


class CourseAssignmentCollaborationSerializer(serializers.ModelSerializer):
    """Serializer for the CourseAssignmentCollaboration model."""

    class Meta:
        """Meta options for CourseAssignmentCollaborationSerializer.

        Attributes:
            model: The CourseAssignmentCollaboration model.
            fields: All model fields.
        """

        model = CourseAssignmentCollaboration
        fields = "__all__"


class StudentsSerializer(serializers.ModelSerializer):
    """Serializer for the Students model."""

    class Meta:
        """Meta options for StudentsSerializer.

        Attributes:
            model: The Students model.
            fields: All model fields.
        """

        model = Students
        fields = "__all__"


class StudentEnrollmentsSerializer(serializers.ModelSerializer):
    """Serializer for the StudentEnrollments model."""

    class Meta:
        """Meta options for StudentEnrollmentsSerializer.

        Attributes:
            model: The StudentEnrollments model.
            fields: All model fields.
        """

        model = StudentEnrollments
        fields = "__all__"


class ProfessorsSerializer(serializers.ModelSerializer):
    """Serializer for the Professors model."""

    class Meta:
        """Meta options for ProfessorsSerializer.

        Attributes:
            model: The Professors model.
            fields: All model fields.
        """

        model = Professors
        fields = "__all__"


class ProfessorEnrollmentsSerializer(serializers.ModelSerializer):
    """Serializer for the ProfessorEnrollments model."""

    class Meta:
        """Meta options for ProfessorEnrollmentsSerializer.

        Attributes:
            model: The ProfessorEnrollments model.
            fields: All model fields.
        """

        model = ProfessorEnrollments
        fields = "__all__"


class TeachingAssistantsSerializer(serializers.ModelSerializer):
    """Serializer for the TeachingAssistants model."""

    class Meta:
        """Meta options for TeachingAssistantsSerializer.

        Attributes:
            model: The TeachingAssistants model.
            fields: All model fields.
        """

        model = TeachingAssistants
        fields = "__all__"


class TeachingAssistantEnrollmentSerializer(serializers.ModelSerializer):
    """Serializer for the TeachingAssistantEnrollment model."""

    class Meta:
        """Meta options for TeachingAssistantEnrollmentSerializer.

        Attributes:
            model: The TeachingAssistantEnrollment model.
            fields: All model fields.
        """

        model = TeachingAssistantEnrollment
        fields = "__all__"
