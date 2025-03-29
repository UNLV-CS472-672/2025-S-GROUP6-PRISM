"""Permissions and role-based access control for the users app."""

from rest_framework.permissions import BasePermission
import logging

# Create a logger for this module.
logger = logging.getLogger(__name__)


def log_role_access(user, action, resource):
    """Log access of a user performing an action on a resource.

    Args:
        user (str): The username or identifier of the user performing the action.
        action (str): The action performed (e.g., "read", "write", "delete").
        resource (str): The target resource of the action.
    """
    logger.info(f"{user} performed {action} on {resource}")


def is_professor(user):
    """Check if the given user belongs to the 'Professor' group.

    Args:
        user (User): The user object to check.

    Returns:
        bool: True if the user is a professor, False otherwise.
    """
    return user.groups.filter(name="Professor").exists()


def is_ta(user):
    """Check if the given user belongs to the 'TA' group.

    Args:
        user (User): The user object to check.

    Returns:
        bool: True if the user is a TA, False otherwise.
    """
    return user.groups.filter(name="TA").exists()


def is_admin(user):
    """Check if a user has administrative privileges.

    Args:
        user (User): The user object to check.

    Returns:
        bool: True if the user is a staff member or a superuser, False otherwise.
    """
    return user.is_staff or user.is_superuser


class IsProfessorOrTA(BasePermission):
    """Allow access if the user is a professor or a TA."""

    def has_permission(self, request, view):
        """Check if the user is a professor or TA.

        Args:
            request (HttpRequest): The incoming request.
            view (View): The view being accessed.

        Returns:
            bool: True if the user is authenticated and is a professor or TA.
        """
        return (
            request.user and request.user.is_authenticated
            and (is_professor(request.user) or is_ta(request.user))
        )


class IsProfessorOrAdmin(BasePermission):
    """Allow access if the user is a professor or an admin."""

    def has_permission(self, request, view):
        """Check if the user is a professor or admin.

        Args:
            request (HttpRequest): The incoming request.
            view (View): The view being accessed.

        Returns:
            bool: True if the user is authenticated and is a professor or admin.
        """
        return (
            request.user
            and request.user.is_authenticated
            and (is_professor(request.user) or is_admin(request.user))
        )


class IsProfessor(BasePermission):
    """Allow access only to professors."""

    def has_permission(self, request, view):
        """Check if the user is a professor.

        Args:
            request (HttpRequest): The incoming request.
            view (View): The view being accessed.

        Returns:
            bool: True if the user is authenticated and is a professor.
        """
        return (
            request.user
            and request.user.is_authenticated
            and is_professor(request.user)
        )


class IsAdmin(BasePermission):
    """Allow access only to admin users."""

    def has_permission(self, request, view):
        """Check if the user has admin privileges.

        Args:
            request (HttpRequest): The incoming request.
            view (View): The view being accessed.

        Returns:
            bool: True if the user is authenticated and is an admin.
        """
        return request.user and request.user.is_authenticated and is_admin(request.user)
