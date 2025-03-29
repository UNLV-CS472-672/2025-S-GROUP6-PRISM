"""User model and manager definitions."""

from django.contrib.auth.models import (
    AbstractUser,
    Group,
    PermissionsMixin,
    BaseUserManager,
    Permission,
)
from django.db import models


class UserManager(BaseUserManager):
    """Custom manager for the User model.

    Provides methods for creating regular users and superusers.

    Methods:
        create_user(email, password, **extra_fields):
            Creates and returns a regular user with the given email and password.
        create_superuser(email, password, **extra_fields):
            Creates and returns a superuser with elevated permissions.
    """

    def create_user(self, email, password, **extra_fields):
        """Create and return a new user with the given email and password.

        If no `username` is provided in extra_fields, it defaults to the email.

        Args:
            email (str): The user's email address.
            password (str): The raw password for the user.
            **extra_fields: Any additional fields to set on the user.

        Raises:
            ValueError: If no email is provided.

        Returns:
            User: The created user instance.
        """
        if not email:
            raise ValueError("Users must have an email address")

        if not extra_fields.get("username"):
            extra_fields["username"] = email

        user = self.model(email=self.normalize_email(email), **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, **extra_fields):
        """Create and return a superuser with elevated privileges.

        Superusers have admin, staff, and superuser permissions.

        Args:
            email (str): The superuser's email address.
            password (str): The raw password for the superuser.
            **extra_fields: Additional fields for the superuser.

        Returns:
            User: The created superuser instance.
        """
        user = self.create_user(email, password, **extra_fields)
        user.is_admin = True
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class User(AbstractUser, PermissionsMixin):
    """Custom user model extending Django's AbstractUser.

    Attributes:
        email (EmailField): Unique email used for authentication.
        first_name (CharField): First name of the user.
        last_name (CharField): Last name of the user.
        is_admin (BooleanField): Indicates admin privileges.
        objects (UserManager): Custom manager for user creation.
        USERNAME_FIELD (str): Field used for authentication (email).
        REQUIRED_FIELDS (list): Required fields other than USERNAME_FIELD.
        groups (ManyToManyField): Custom relationship to Group model.
        user_permissions (ManyToManyField): Custom relationship to Permission model.

    Methods:
        __str__(): Return string representation of the user.
    """

    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    is_admin = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["first_name", "last_name"]

    groups = models.ManyToManyField(
        Group,
        related_name="custom_user_set",
        blank=True,
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name="custom_user_permissions_set",
        blank=True,
    )

    def __str__(self):
        """Return a string representation of the user.

        Format: "FirstName LastName (email@example.com)".

        Returns:
            str: Formatted string representing the user.
        """
        return f"{self.first_name} {self.last_name} ({self.email})"
