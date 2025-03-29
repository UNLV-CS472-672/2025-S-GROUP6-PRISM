"""Tests for the User model."""

from django.test import TestCase
from django.contrib.auth import get_user_model
from django.db import IntegrityError


class UserModelTest(TestCase):
    """Test suite for the User model.

    Includes tests for:
    - User creation and validation
    - Email normalization
    - Superuser creation
    - Field updates and constraints
    """

    def test_create_user_successfully(self):
        """Create a user with valid input and verify attributes."""
        email = "test@example.com"
        password = "123test"
        first_name = "testName"
        last_name = "testLast"
        user = get_user_model().objects.create_user(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
        )
        self.assertEqual(user.email, email)
        self.assertEqual(user.first_name, first_name)
        self.assertEqual(user.last_name, last_name)
        self.assertTrue(user.check_password(password))

    def test_new_user_email_normalization(self):
        """Normalize email domain on user creation."""
        dummy_emails = [
            ["test1@EXAMPLE.com", "test1@example.com"],
            ["Test2@Example.com", "Test2@example.com"],
            ["TEST3@EXAMPLE.COM", "TEST3@example.com"],
            ["test4@example.COM", "test4@example.com"],
            ["test5@ExamPle.Com", "test5@example.com"],
        ]
        for email, expected in dummy_emails:
            user = get_user_model().objects.create_user(
                email=email, password="sample123"
            )
            self.assertEqual(user.email, expected)

    def test_new_user_without_email_error(self):
        """Raise ValueError if email is not provided."""
        with self.assertRaises(ValueError):
            get_user_model().objects.create_user(
                email="",
                password="sample123",
                first_name="first",
                last_name="last",
            )

    def test_create_superuser(self):
        """Create a superuser and verify permissions."""
        user = get_user_model().objects.create_superuser(
            email="test@example.com",
            password="test123",
            first_name="first",
            last_name="last",
        )
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.is_admin)

    def test_update_user_fields(self):
        """Update a user’s first and last name and save changes."""
        user = get_user_model().objects.create_user(
            email="update@example.com",
            password="pass123",
            first_name="Initial",
            last_name="Name",
        )
        user.first_name = "Updated"
        user.last_name = "User"
        user.save()
        user.refresh_from_db()
        self.assertEqual(user.first_name, "Updated")
        self.assertEqual(user.last_name, "User")

    def test_unique_email_constraint(self):
        """Raise IntegrityError for duplicate user email."""
        email = "unique@example.com"
        get_user_model().objects.create_user(
            email=email,
            password="pass123",
            first_name="Unique",
            last_name="User",
        )
        with self.assertRaises(IntegrityError):
            get_user_model().objects.create_user(
                email=email,
                password="pass456",
                first_name="Duplicate",
                last_name="User",
            )

    def test_minimal_valid_input_defaults(self):
        """Create a user with minimal input and verify defaults."""
        user = get_user_model().objects.create_user(
            email="minimal@example.com",
            password="minpass",
            first_name="Min",
            last_name="User",
        )
        self.assertTrue(user.is_active)
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)
