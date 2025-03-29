"""Expanded tests for the User API."""

from django.test import TestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status


# A helper to generate the base URL for the user list endpoint (ModelViewSet).
USER_LIST_URL = reverse("user:user-list")


def detail_user_url(user_id):
    """Generate the URL for the detail view of a specific user.

    Args:
        user_id (int): The ID of the user for which the detail URL is generated.

    Returns:
        str: The URL string for the user's detail view.
    """
    return reverse("user:user-detail", args=[user_id])


def create_user(**params):
    """Create and return a new user."""
    return get_user_model().objects.create_user(**params)


class PublicUserApiTests(TestCase):
    """Unit tests for the Public User API."""

    def setUp(self):
        """Set up the test client for API requests."""
        self.client = APIClient()

    def test_create_user_success(self):
        """Test creating a user is successful."""
        payload = {
            "email": "test@example.com",
            "password": "sample123",
            "first_name": "Dummy",
            "last_name": "lastDummy",
        }
        res = self.client.post(USER_LIST_URL, payload)
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        user = get_user_model().objects.get(email=payload["email"])
        self.assertTrue(user.check_password(payload["password"]))
        self.assertNotIn("password", res.data)

    def test_user_with_email_exist_error(self):
        """Test error returned if user with email already exists."""
        payload = {
            "email": "test@example.com",
            "password": "sample123",
            "first_name": "Dummy",
            "last_name": "lastDummy",
        }
        create_user(**payload)
        res = self.client.post(USER_LIST_URL, payload)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    def test_password_too_short_error(self):
        """Test an error is returned if password is less than 5 characters."""
        payload = {
            "email": "test@example.com",
            "password": "meh",
            "first_name": "Dummy",
            "last_name": "lastDummy",
        }
        res = self.client.post(USER_LIST_URL, payload)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

        user_exists = get_user_model().objects.filter(email=payload["email"]).exists()
        self.assertFalse(user_exists)

    def test_create_user_also_creates_professor(self):
        """Test creating a user also creates a corresponding Professor."""
        payload = {
            "email": "professor@example.com",
            "password": "prof123",
            "first_name": "Prof",
            "last_name": "Test",
        }
        res = self.client.post(USER_LIST_URL, payload)
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)

        user = get_user_model().objects.get(email=payload["email"])
        from courses.models import Professor

        professor_exists = Professor.objects.filter(user=user).exists()
        self.assertTrue(professor_exists)

    def test_list_users(self):
        """Test listing users (GET /user/)."""
        user1 = create_user(
            email="list1@example.com",
            password="pass123",
            first_name="List",
            last_name="UserOne",
        )
        user2 = create_user(
            email="list2@example.com",
            password="pass123",
            first_name="List",
            last_name="UserTwo",
        )
        res = self.client.get(USER_LIST_URL)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 2)

        emails = [u["email"] for u in res.data]
        self.assertIn(user1.email, emails)
        self.assertIn(user2.email, emails)

    def test_filter_users_by_email(self):
        """Test filtering user list by email."""
        user1 = create_user(
            email="filterme@example.com",
            password="pass123",
            first_name="Filter",
            last_name="Test",
        )
        create_user(
            email="another@example.com",
            password="pass123",
            first_name="Other",
            last_name="User",
        )
        url = f"{USER_LIST_URL}?email=filterme"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["email"], user1.email)

    def test_filter_users_by_first_name(self):
        """Test filtering user list by first_name."""
        user1 = create_user(
            email="somebody@example.com",
            password="pass123",
            first_name="UniqueName",
            last_name="Filter",
        )
        create_user(
            email="someoneelse@example.com",
            password="pass123",
            first_name="Common",
            last_name="Name",
        )
        url = f"{USER_LIST_URL}?first_name=UniqueName"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["first_name"], user1.first_name)

    def test_filter_users_by_last_name(self):
        """Test filtering user list by last_name."""
        user1 = create_user(
            email="lastfilter@example.com",
            password="pass123",
            first_name="LastFilter",
            last_name="Candidate",
        )
        create_user(
            email="other@example.com",
            password="pass123",
            first_name="Other",
            last_name="Name",
        )
        url = f"{USER_LIST_URL}?last_name=Candidate"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]["email"], user1.email)

    def test_order_users_by_first_name(self):
        """Test ordering user list by first name."""
        user_a = create_user(
            email="a@example.com",
            password="pass123",
            first_name="Aaa",
            last_name="Smith",
        )
        user_b = create_user(
            email="b@example.com",
            password="pass123",
            first_name="Bbb",
            last_name="Smith",
        )
        url = f"{USER_LIST_URL}?ordering=-first_name"
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data[0]["first_name"], user_b.first_name)
        self.assertEqual(res.data[1]["first_name"], user_a.first_name)

    def test_retrieve_user_success(self):
        """Test retrieving a single user by ID."""
        user = create_user(
            email="retrieve@example.com",
            password="pass123",
            first_name="Retrieve",
            last_name="User",
        )
        url = detail_user_url(user.id)
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data["email"], user.email)
        self.assertEqual(res.data["first_name"], user.first_name)
        self.assertEqual(res.data["last_name"], user.last_name)

    def test_retrieve_user_not_found(self):
        """Test retrieving a non-existent user returns 404."""
        url = detail_user_url(999999)
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", res.data)

    def test_update_user_not_found(self):
        """Test updating a non-existent user returns 404."""
        payload = {
            "email": "doesntmatter@example.com",
            "first_name": "Nope",
            "last_name": "Nope",
        }
        url = detail_user_url(999999)
        res = self.client.put(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_404_NOT_FOUND)

    def test_update_user_missing_required_field(self):
        """Test updating a user missing required field returns 400."""
        user = create_user(
            email="required@example.com",
            password="pass123",
            first_name="Req",
            last_name="Test",
        )
        url = detail_user_url(user.id)
        payload = {
            "first_name": "NoEmailProvided",
            "last_name": "MissingEmail",
        }
        res = self.client.put(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("email", str(res.data))

    def test_partial_update_user_success(self):
        """Test partially updating a user with PATCH."""
        user = create_user(
            email="patchme@example.com",
            password="pass123",
            first_name="Partial",
            last_name="Update",
        )
        url = detail_user_url(user.id)
        payload = {"first_name": "PatchedName"}
        res = self.client.patch(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        user.refresh_from_db()
        self.assertEqual(user.first_name, "PatchedName")

    def test_partial_update_user_duplicate_email(self):
        """Test partial update with an email that already exists."""
        user1 = create_user(
            email="duplicate1@example.com",
            password="pass123",
            first_name="Dupe1",
            last_name="Test",
        )
        create_user(
            email="duplicate2@example.com",
            password="pass123",
            first_name="Dupe2",
            last_name="Test",
        )
        url = detail_user_url(user1.id)
        payload = {"email": "duplicate2@example.com"}
        res = self.client.patch(url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("email", str(res.data))

    def test_delete_user(self):
        """Test deleting a user (or return 405 if not allowed)."""
        user = create_user(email="delete@example.com", password="pass123")
        url = detail_user_url(user.id)
        res = self.client.delete(url)
        if res.status_code == status.HTTP_204_NO_CONTENT:
            self.assertFalse(get_user_model().objects.filter(id=user.id).exists())
        else:
            self.assertEqual(res.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)
