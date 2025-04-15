"""Created by Eli Rosales, 3/24/2025.

This is the test script for "extract_student_data_from_API.py"

Uses unittest.mock to mock the outputs you would get form the codegrade
API as we do not want to send requests for these tests to the actual
codegrade servers.

Verifys:
    - most getter funcitons
    - good and bad zip files
    - extract_all_assignments()
    - extract_csv()
    - delete_created_folder()
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
from data_ingestion.extract_student_data_from_API import API_Data, main
import io
from io import StringIO
import os
import zipfile
import httpx
import datetime


class TestAPIData(unittest.TestCase):
    """Test class for 'extract student data from API'."""

    def setUp(self):
        """Set up client and class object."""
        self.mock_client = MagicMock()
        # assignment = MagicMock()
        # self.mock_client.assignment = assignment
        self.api_data = API_Data(self.mock_client)

    def create_mock_assignment(
        self, assignment_id, name, lock_date=None, deadline=None, max_grade=100
    ):
        """Create a mock assignment."""
        assignment = MagicMock()
        assignment.id = assignment_id
        assignment.name = name
        assignment.lock_date = lock_date
        assignment.deadline = deadline
        assignment.max_grade = max_grade
        return assignment

    def create_mock_submission(
        self, submission_id, user_id, user_name, user_username, grade, group_name=None
    ):
        """Create a mock submission with all needed attributes."""
        submission = MagicMock()
        submission.id = submission_id
        submission.user = MagicMock()
        submission.user.id = user_id
        submission.user.name = user_name
        submission.user.username = user_username
        submission.grade = grade
        submission.user.group = MagicMock() if group_name else None
        if group_name:
            submission.user.group.name = group_name
        return submission

    # start testing
    def test_handle_maybe(self):
        """Test that we can extract data from input."""
        maybe_mock = MagicMock()
        maybe_mock.try_extract.return_value = "extracted value"
        result = self.api_data.handle_maybe(maybe_mock)
        self.assertEqual(result, "extracted value")

    @patch("data_ingestion.extract_student_data_from_API.os")
    def test_mkdir_success(self, mock_os):
        """Test mkdirs returns true and only called once."""
        mock_os.makedirs.return_value = None
        result = self.api_data.mkdir("test_dir")
        self.assertTrue(result)
        mock_os.makedirs.assert_called_once_with("test_dir", exist_ok=True)

    @patch("data_ingestion.extract_student_data_from_API.os")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_mkdir_failure(self, mock_stderr, mock_os):
        """Exception handling and returns an error."""
        mock_os.makedirs.side_effect = Exception("mkdir failed")
        result = self.api_data.mkdir("test_dir")
        self.assertFalse(result)
        self.assertIn("mkdir failed", mock_stderr.getvalue())

    def test_get_course(self):
        """Test the get_course method."""
        mock_course1 = MagicMock()
        mock_course1.id = "1"
        mock_course1.name = "Dev course"
        mock_course1.created_at = "may 12"
        mock_course2 = MagicMock()
        mock_course2.id = "2"
        mock_course2.name = "cs 101"
        mock_course2.created_at = "january 12"
        mock_courses_list = [mock_course1, mock_course2]
        self.mock_client.course.get_all.return_value = mock_courses_list
        course = self.api_data.get_course(self.mock_client)
        self.assertEqual(course, mock_course1)

    def test_get_course_fail(self):
        """Test the fail case."""
        self.mock_client.course.get_all.return_value = Exception()
        result = self.api_data.get_course(self.mock_client)
        self.assertIsInstance(result, Exception)

    def test_get_assignments(self):
        """Test return of all assignments."""
        mock_course = MagicMock()
        mock_assignments = ["assignment1", "assignment2"]
        mock_course.assignments = mock_assignments
        self.api_data.course = mock_course
        assignments = self.api_data.get_assignments()
        self.assertEqual(assignments, mock_assignments)

    def test_get_course_info(self):
        """Test we get the wanted course info."""
        mock_course = MagicMock()
        mock_course.id = "course_id"
        mock_course.name = "course_name"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        course_info = self.api_data.get_course_info()
        expected_info = {
            "Course-ID": "course_id",
            "Name": "course_name",
            "Created-Date": "created_date",
        }
        self.assertEqual(course_info, expected_info)

    def test_get_rubric_grades_dict_no_assignments(self):
        """
        Test case: No assignments are provided.

        Expected behavior: Should return an empty dictionary.
        """
        mock_assignments = []
        mock_result = self.api_data.get_rubric_grades_dict(mock_assignments)
        self.assertEqual(mock_result, {})

    def test_get_all_submissions_success(self):
        """Test for retrieving all submissions from an assignment."""
        assignment = self.create_mock_assignment("1234", "Test Assignment")
        self.mock_client.assignment = assignment
        self.mock_client.assignment.get_all_submissions.return_value = [
            "submission1",
            "submission2",
        ]
        submissions = self.api_data.get_all_submissions(assignment)
        self.assertEqual(submissions, ["submission1", "submission2"])
        self.mock_client.assignment.get_all_submissions.assert_called_once_with(
            assignment_id="1234"
        )

    def test_get_all_submissions_fail(self):
        """We get an exception from api call get_all_submissions."""
        assignment = self.create_mock_assignment("1234", "Test Assignment")
        self.mock_client.assignment = assignment
        self.mock_client.assignment.get_all_submissions.side_effect = Exception(
            "invalid Assignment input"
        )
        result = self.api_data.get_all_submissions(assignment)
        self.assertFalse(result)

    def test_get_all_graders(self):
        """Test to get all graders from assignment."""
        assignment = self.create_mock_assignment("assignment1", "Test Assignment")
        self.mock_client.assignment.get_all_graders.return_value = [
            "grader1",
            "grader2",
        ]
        graders = self.api_data.get_all_graders(assignment)
        self.assertEqual(graders, ["grader1", "grader2"])
        self.mock_client.assignment.get_all_graders.assert_called_once_with(
            assignment_id="assignment1"
        )

    def test_get_rubric(self):
        """Test rubric from api call."""
        assignment = self.create_mock_assignment("1234", "Test Assignment")
        mock_rubric_data = {"rubric_item": "value"}
        self.mock_client.assignment.get_rubric.return_value = mock_rubric_data
        rubric = self.api_data.get_rubric(assignment)
        self.assertEqual(rubric, mock_rubric_data)
        self.mock_client.assignment.get_rubric.assert_called_with(assignment_id="1234")

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_rubric_failure(self, stdout):
        """
        Test case: Method spits an Exeption.

        Expected behavior: Shoukd return "COUNT NOT FIND RUBRIC" in iostream.
        """
        fail_string = "COULD NOT FIND RUBRIC"
        self.mock_client.assignment.get_rubric.side_effect = Exception(fail_string)
        assignment = self.create_mock_assignment("1", "Test Assignment")
        result = self.api_data.get_rubric(assignment)
        self.assertFalse(result)
        self.assertIn(
            fail_string,
            stdout.getvalue().strip(),
        )

    def test_get_desc_success(self):
        """
        Test case: Get description was a success.

        Expected behavior: Should return the description.
        """
        assignment = self.create_mock_assignment("1", "Test Assignment")
        mock_desc = "Test Desciption"
        self.mock_client.assignment.get_description.return_value = mock_desc
        result = self.api_data.get_desc(assignment)
        self.assertEqual(result, mock_desc)

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_desc_failure(self, stdout):
        """
        Test case: Method throws an Exception.

        Expected behavior: Should return "FAILED TO GET DESCRIPTION".
        """
        fail_string = "FAILED TO GET DESCRIPTION"
        self.mock_client.assignment.get_description.side_effect = Exception(fail_string)
        assignment = self.create_mock_assignment("1", "Test Assignment")
        result = self.api_data.get_desc(assignment)
        self.assertFalse(result)
        self.assertIn(
            fail_string,
            stdout.getvalue().strip(),
        )

    def test_get_time_frames(self):
        """
        Test case: Test time frames gives correct (given) output.

        Expected behavior: Should return an given output.
        """
        assignment = self.create_mock_assignment("1", "Test Assignment")
        mock_time_frame = "may - june"
        self.mock_client.assignment.get_timeframes.return_value = mock_time_frame
        result = self.api_data.get_time_frames(assignment)
        self.assertEqual(result, mock_time_frame)

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_time_frames_failure(self, stdout):
        """
        Test case: Method Throws Exception.

        Expected behavior: Should return "FAILED TO GET TIME FRAMES".
        """
        assignment = self.create_mock_assignment("1", "Test Assignment")
        fail_string = "FAILED TO GET TIME FRAMES"
        self.mock_client.assignment.get_timeframes.side_effect = Exception(fail_string)
        result = self.api_data.get_time_frames(assignment)
        self.assertFalse(result)
        self.assertIn(
            fail_string,
            stdout.getvalue().strip(),
        )

    def test_get_feedback(self):
        """
        Test case: Get feedback for assignment.

        Expected behavior: Should return "feedback 1".
        """
        assignment = self.create_mock_assignment("1", "Test Assignment")
        mock_feedback = "feedback 1"
        self.mock_client.assignment.get_all_feedback.return_value = mock_feedback
        result = self.api_data.get_feedback(assignment)
        self.assertEqual(result, mock_feedback)

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_feedback_failure(self, stdout):
        """
        Test case: Method throws Exception.

        Expected behavior: Should return "Could not find feedback".
        """
        assignment = self.create_mock_assignment("1", "Test Assignment")
        fail_string = "Could not find feedback"
        self.mock_client.assignment.get_all_feedback.side_effect = Exception(
            fail_string
        )
        result = self.api_data.get_feedback(assignment)
        self.assertFalse(result)
        self.assertIn(
            fail_string,
            stdout.getvalue().strip(),
        )

    def test_get_users(self):
        """
        Test case: Test get users from a course.

        Expected behavior: Should return list of users.
        """
        mock_course = MagicMock()
        mock_course.id = "course_id"
        mock_course.name = "course_name"
        mock_course.created_at = "created_date"
        users_list = ["user1", "user2"]
        self.mock_client.course.get_all_users.return_value = users_list
        result = self.api_data.get_users(mock_course)
        self.assertEqual(result, users_list)

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_user_failure(self, stdout):
        """
        Test case: Method throws Exception.

        Expected behavior: Should return "Could not find user".
        """
        mock_course = MagicMock()
        fail_string = "Could not find user"
        self.mock_client.course.get_all_users.side_effect = Exception(fail_string)
        result = self.api_data.get_users(mock_course)
        self.assertFalse(result)
        self.assertIn(
            fail_string,
            stdout.getvalue().strip(),
        )

    def test_get_all_user_submissions(self):
        """
        Test case: Test get all user submissions success.

        Expected behavior: Should return submissions of a given user.
        """
        mock_course = MagicMock()
        mock_course.id = "course_id"
        mock_course.name = "course_name"
        mock_course.created_at = "created_date"
        mock_submission1 = self.create_mock_submission(
            "sub1", "user1ID", "User One", "user1", 90
        )
        mock_submission2 = self.create_mock_submission(
            "sub2", "user1ID", "User One", "user1", 100
        )
        mock_submission_list = [mock_submission1, mock_submission2]
        self.mock_client.course.get_submissions_by_user.return_value = (
            mock_submission_list
        )
        result = self.api_data.get_all_user_submissions(mock_course, "user1ID")
        self.assertEqual(result, mock_submission_list)

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_all_user_submissions_failure(self, stdout):
        """
        Test case: Method Throws Exception.

        Expected behavior: Should return "Could not find submissions".
        """
        mock_course = MagicMock()
        fail_string = "Could not find submissions"
        self.mock_client.course.get_submissions_by_user.side_effect = Exception(
            fail_string
        )
        result = self.api_data.get_all_user_submissions(mock_course, "1234")
        self.assertFalse(result)
        self.assertIn(
            fail_string,
            stdout.getvalue().strip(),
        )

    def test_get_rubric_grade(self):
        """Test formatting of get rubric grade method."""
        mock_grade_data = {"grade_item": "value"}
        self.mock_client.submission.get_rubric_result.return_value = mock_grade_data
        grade = self.api_data.get_rubric_grade("sub123")
        self.assertEqual(grade, mock_grade_data)
        self.mock_client.submission.get_rubric_result.assert_called_with(
            submission_id="sub123"
        )

    def test_get_rubric_grade_none(self):
        """Test the error handleing for a submission wo a rubric."""
        self.mock_client.submission.get_rubric_result.side_effect = Exception(
            "Rubric not found"
        )
        grade = self.api_data.get_rubric_grade("sub123")
        self.assertIsNone(grade)
        self.mock_client.submission.get_rubric_result.assert_called_with(
            submission_id="sub123"
        )

    @patch("data_ingestion.extract_student_data_from_API.os")
    @patch("data_ingestion.extract_student_data_from_API.json")
    def test_get_json_file_success(self, mock_json, mock_os):
        """Test json is called and we create the output file .cg-info.json."""
        m = mock_open()
        with patch("builtins.open", m):
            mock_os.path.join.return_value = "test_path/.cg-info.json"
            mock_os.makedirs.return_value = None
            test_dict = {"key": "value"}
            self.api_data.get_json_file(test_dict, "test_path")
            mock_json.dump.assert_called()
            mock_os.makedirs.assert_called_once_with("test_path", exist_ok=True)

    @patch("data_ingestion.extract_student_data_from_API.os")
    @patch("data_ingestion.extract_student_data_from_API.json")
    def test_get_json_file_mkdir_failure(self, mock_json, mock_os):
        """Exception handling."""
        mock_os.makedirs.side_effect = Exception("mkdir failed")
        test_dict = {"key": "value"}
        self.api_data.get_json_file(test_dict, "test_path")
        mock_os.makedirs.assert_called_once_with("test_path", exist_ok=True)
        mock_json.dump.assert_not_called()

    @patch("data_ingestion.extract_student_data_from_API.codegrade")
    @patch("data_ingestion.extract_student_data_from_API.os")
    @patch("data_ingestion.extract_student_data_from_API.zipfile")
    @patch("data_ingestion.extract_student_data_from_API.io")
    @patch("data_ingestion.extract_student_data_from_API.shutil")
    def test_download_submission_success_individual(
        self, mock_shutil, mock_io, mock_zipfile, mock_os, mock_codegrade
    ):
        """Test that we can doanload a submission and that output file was created."""
        mock_submission = self.create_mock_submission(
            "sub1", "user1", "User One", "user1", 90
        )
        mock_zipinfo = MagicMock(name="test.zip")
        mock_zipdata = b"mock zip data"
        self.mock_client.submission.get.return_value = mock_zipinfo
        mock_codegrade.file.download.return_value = mock_zipdata
        mock_os.path.join.side_effect = lambda *args: os.path.join(*args)
        mock_os.path.basename.return_value = "file.txt"
        mock_zip_file_mock = MagicMock()
        mock_zipfile.ZipFile.return_value.__enter__.return_value = mock_zip_file_mock
        mock_zip_file_mock.namelist.return_value = ["path/to/file.txt"]
        mock_source_file = MagicMock()
        mock_zip_file_mock.open.return_value = mock_source_file
        mock_target_file = MagicMock()
        mock_io.BytesIO.return_value = MagicMock()
        mock_os.makedirs.return_value = None
        mock_os.path.exists.return_value = False
        mock_shutil = MagicMock()
        mock_shutil.copyfileobj.return_value = None
        with patch("builtins.open", mock_target_file):
            self.api_data.download_submission(mock_submission, "output_dir")
        mock_os.makedirs.assert_called_once_with(
            os.path.join("output_dir", "User One"), exist_ok=True
        )

    @patch("data_ingestion.extract_student_data_from_API.codegrade")
    @patch("data_ingestion.extract_student_data_from_API.os")
    @patch("data_ingestion.extract_student_data_from_API.zipfile")
    @patch("data_ingestion.extract_student_data_from_API.io")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_download_submission_bad_zip(
        self, mock_stderr, mock_io, mock_zipfile, mock_os, mock_codegrade
    ):
        """Test that the zipfile spits an error."""
        mock_submission = self.create_mock_submission(
            "sub1", "user1", "User One", "user1", 90
        )
        mock_zipinfo = MagicMock(name="test.zip")
        mock_zipdata = b"mock zip data"
        self.mock_client.submission.get.return_value = mock_zipinfo
        mock_codegrade.file.download.return_value = mock_zipdata
        mock_os.path.join.side_effect = lambda *args: os.path.join(*args)
        mock_os.makedirs.return_value = None
        # mock_zipfile.ZipFile.side_effect = zipfile.BadZipFile("Bad zip file")
        mock_zip_file_mock = MagicMock()
        mock_zipfile.ZipFile.return_value.__enter__.return_value = mock_zip_file_mock
        mock_zip_file_mock.namelist.side_effect = zipfile.BadZipFile("Bad zip file")
        mock_io.BytesIO.return_value = MagicMock()
        self.api_data.download_submission(mock_submission, "output_dir")
        self.assertIn("Invalid zip file", mock_stderr.getvalue())

    def test_download_submission_httpx_read_error_retry(self):
        """Test our exception case works."""
        mock_submission = self.create_mock_submission(
            "sub1", "user1", "User One", "user1", 90
        )
        self.mock_client.file.download.side_effect = httpx.ReadError("Read error")
        with self.assertRaises(httpx.ReadError):
            self.api_data.download_submission(mock_submission, "output_dir")

    @patch("data_ingestion.extract_student_data_from_API.os")
    @patch("data_ingestion.extract_student_data_from_API.API_Data.mkdir")
    def test_download_submission_mkdir_failure(self, mock_mkdir, mock_os):
        """
        Test case: Method Throws Exception.

        Expected behavior: Should return empty variable.
        """
        mock_submission = self.create_mock_submission(
            "sub1", "user1", "User One", "user1", 90
        )
        mock_zipinfo = MagicMock(name="test.zip")
        mock_zipdata = b"mock zip data"
        self.mock_client.submission.get.return_value = mock_zipinfo
        self.mock_client.file.download.return_value = mock_zipdata
        mock_mkdir.return_value = None
        result = self.api_data.download_submission(mock_submission, "output_dir")
        self.assertFalse(result)

    @patch("data_ingestion.extract_student_data_from_API.os")
    @patch("data_ingestion.extract_student_data_from_API.json")
    @patch("data_ingestion.extract_student_data_from_API.datetime")
    @patch("data_ingestion.extract_student_data_from_API.API_Data.get_json_file")
    @patch("data_ingestion.extract_student_data_from_API.API_Data.make_zip_archive")
    @patch("data_ingestion.extract_student_data_from_API.API_Data.download_submission")
    def test_extract_all_assignments_success(
        self, mock_down, mock_zip, mock_json_file, mock_datetime, mock_json, mock_os
    ):
        """Mock course and assignments."""
        mock_course = MagicMock()
        mock_course.id = "course_id"
        mock_course.name = "course_name"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        mock_now = datetime.datetime.now()
        mock_datetime.datetime.now.return_value = mock_now
        mock_lock = datetime.datetime(2024, 5, 24, 0, 0, 0)
        mock_datetime.datetime.return_value = mock_lock
        self.assertGreater(mock_now, mock_lock)
        mock_assignment1 = self.create_mock_assignment(
            assignment_id=int(1001), name="Assignment One", lock_date=mock_lock
        )
        mock_assignment2 = self.create_mock_assignment(
            assignment_id=int(1002), name="Assignment Two", lock_date=mock_lock
        )
        mock_assignments = [mock_assignment1, mock_assignment2]
        mock_submission1 = self.create_mock_submission(
            "sub1", "user1", "User One", "user1", 90
        )
        mock_submission2 = self.create_mock_submission(
            "sub1", "user1", "User One", "user1", 90
        )
        mock_submissions = [mock_submission1, mock_submission2]
        # Mock os and json behavior
        mock_os.path.join.side_effect = lambda *args: os.path.join(*args)
        mock_os.makedirs.return_value = None
        mock_json.dump.return_value = None
        self.mock_client.assignment.get_all_submissions.return_value = mock_submissions
        mock_down.return_value = None
        self.api_data.extract_all_assignments(mock_assignments)

        self.assertEqual(
            self.api_data.download_submission.call_count, 4
        )  # 2 each assignment
        mock_json_file.assert_called()
        mock_zip.assert_called()

    def test_extract_all_assignments_no_assignments(self):
        """Test with no assignments given."""
        mock_course = MagicMock()
        mock_course.id = "course_id"
        mock_course.name = "course_name"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        mock_no_assignments = []
        self.api_data.extract_all_assignments(mock_no_assignments)

    @patch("data_ingestion.extract_student_data_from_API.os")
    def test_extract_all_assignments_mkdir_failure(self, mock_os):
        """Test when mkdir fails."""
        mock_course = MagicMock()
        mock_course.id = "test_course_id"
        mock_course.name = "Test Course Name"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        mock_lock = datetime.datetime(2024, 5, 24, 0, 0, 0)
        mock_assignment = self.create_mock_assignment(
            "assign1", "Assignment One", mock_lock
        )
        mock_assignments = [mock_assignment]
        mock_os.path.join.side_effect = lambda *args: os.path.join(*args)
        mock_os.makedirs.side_effect = OSError("Failed to create directory")

        with self.assertRaises(OSError):
            self.api_data.extract_all_assignments(mock_assignments)

        mock_os.makedirs.assert_called_once()

    @patch("sys.stdout", new_callable=StringIO)
    @patch("data_ingestion.extract_student_data_from_API.datetime")
    def test_extract_all_assignments_lock_date_too_far(self, mock_datetime, stdout):
        """
        Test case: Tests if lock date works correctly.

        Expected behavior:
        Continue if lockdate has not passed yet.
        Should return "not passed yet for {assignment.name}".
        """
        mock_course = MagicMock()
        mock_course.id = "test_course_id"
        mock_course.name = "Test Course Name"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        mock_lock = datetime.datetime(9999, 5, 24, 0, 0, 0)
        mock_datetime.datetime.return_value = mock_lock

        mock_assignment = self.create_mock_assignment(
            "assign1", "Assignment One", mock_lock
        )
        mock_submission1 = self.create_mock_submission(
            "sub1", "user1", "User One", "user1", 90
        )
        mock_submission2 = self.create_mock_submission(
            "sub2", "user2", "User Two", "user2", 100
        )
        mock_assignments = [mock_assignment]
        mock_submissions = [mock_submission1, mock_submission2]

        mock_now = datetime.datetime.now()
        mock_datetime.datetime.now.return_value = mock_now

        self.mock_client.assignment.get_all_submissions.return_value = mock_submissions
        self.api_data.extract_all_assignments(mock_assignments)
        self.assertIn(
            f"not passed yet for {mock_assignment.name}",
            stdout.getvalue().strip(),
        )

    @patch("sys.stdout", new_callable=StringIO)
    @patch("builtins.open", new_callable=mock_open)
    @patch("data_ingestion.extract_student_data_from_API.datetime")
    def test_extract_csv_success(self, mock_datetime, mock_file, stdout):
        """Test the extract_csv method."""
        mock_course = MagicMock()
        mock_course.id = "course_id"
        mock_course.name = "Success_Course"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        mock_lock = datetime.datetime(2024, 5, 24, 0, 0, 0)
        assignment1 = self.create_mock_assignment("1", "Test Assignment 1", mock_lock)
        assignment2 = self.create_mock_assignment("2", "Test Assignment 2", mock_lock)
        assignments = [assignment1, assignment2]
        mock_submission1 = self.create_mock_submission(
            "sub1", "user1", "User One", "user1dn", 66
        )
        mock_submission2 = self.create_mock_submission(
            "sub1", "user2", "User Two", "user2dn", 66
        )
        mock_submissions = [mock_submission1, mock_submission2]
        self.mock_client.assignment.get_all_submissions.return_value = mock_submissions
        mock_datetime.datetime.return_value = mock_lock
        mock_now = datetime.datetime.now()
        mock_datetime.datetime.now.return_value = mock_now
        # Get the time now to check if mock lock date >(after) mock now; no it should not be

        self.api_data.get_output_dir = MagicMock(
            return_value="output/test_course_assignment_1"
        )
        self.api_data.get_feedback = MagicMock()
        self.api_data.get_grade = MagicMock(return_value=66)

        self.api_data.extract_csv(assignments)

        mock_file.assert_called_with(
            "output/test_course_assignment_1.csv", "w", newline=""
        )
        self.assertIn(
            "SUCCESS! CSV file 'output/test_course_assignment_1.csv' created.",
            stdout.getvalue().strip(),
        )

    @patch("sys.stdout", new_callable=StringIO)
    @patch("data_ingestion.extract_student_data_from_API.datetime")
    def test_extract_csv_fail_submissions(self, mock_datetime, stdout):
        """Test the fail cases of extract_csv: no submissions case."""
        # Mock now = datetime.datetime.()
        mock_now = datetime.datetime.now()
        mock_datetime.datetime.now.return_value = mock_now

        # Mock course
        mock_course = MagicMock()
        mock_course.id = "0001"
        mock_course.name = "Failure_Course"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        mock_lock = datetime.datetime(2024, 5, 24, 0, 0, 0)
        assignment1 = self.create_mock_assignment("1", "Test Assignment 1", mock_lock)
        assignment2 = self.create_mock_assignment("2", "Test Assignment 2", mock_lock)
        assignments = [assignment1, assignment2]
        mock_submissions = []
        self.mock_client.assignment.get_all_submissions.return_value = mock_submissions

        self.api_data.extract_csv(assignments)

        self.assertIn("No submissions for Test Assignment 1", stdout.getvalue().strip())
        self.assertIn("No submissions for Test Assignment 2", stdout.getvalue().strip())

    @patch("sys.stdout", new_callable=StringIO)
    @patch("data_ingestion.extract_student_data_from_API.datetime")
    def test_extract_csv_fail_lockdate(self, mock_datetime, stdout):
        """Test failure of extract_csv for lockdate has not past yet."""
        # Mock now = datetime.datetime.()
        mock_now = datetime.datetime.now()
        mock_datetime.datetime.now.return_value = mock_now

        # Mock course
        mock_course = MagicMock()
        mock_course.id = "0001"
        mock_course.name = "Failure_Course_Lockdate"
        mock_course.created_at = "created_date"
        self.api_data.course = mock_course
        # make lockdate in the future
        mock_lock = datetime.datetime(2026, 5, 24, 0, 0, 0)
        mock_datetime.datetime.return_value = mock_lock
        assignment1 = self.create_mock_assignment("1", "Test Assignment 1", mock_lock)
        assignment2 = self.create_mock_assignment("2", "Test Assignment 2", mock_lock)
        assignments = [assignment1, assignment2]
        mock_submission1 = self.create_mock_submission(
            "sub1", "user1", "User One", "user1dn", 66
        )
        mock_submissions = [mock_submission1]
        self.mock_client.assignment.get_all_submissions.return_value = mock_submissions

        self.api_data.extract_csv(assignments)

        self.assertIn(
            f"Lock date ({mock_lock}) not passed yet for Test Assignment 1",
            stdout.getvalue().strip(),
        )
        self.assertIn(
            f"Lock date ({mock_lock}) not passed yet for Test Assignment 2",
            stdout.getvalue().strip(),
        )

    @patch("sys.stdout", new_callable=StringIO)
    @patch("data_ingestion.extract_student_data_from_API.shutil")
    def test_delete_created_folder_success(self, mock_shutil, stdout):
        """Test the deletion of a folder."""
        self.api_data.create_folder_path = "output/cg_data"
        mock_shutil.rmtree.return_value = None
        self.api_data.delete_created_folder()
        self.assertEqual(
            "Directory 'output/cg_data' deleted successfully.",
            stdout.getvalue().strip(),
        )

    @patch("sys.stdout", new_callable=StringIO)
    @patch("data_ingestion.extract_student_data_from_API.shutil")
    def test_delete_created_folder_failure_file_not_found(self, mock_shutil, stdout):
        """Test the file not found error."""
        self.api_data.create_folder_path = "output/cg_data"
        mock_shutil.rmtree.side_effect = FileNotFoundError()
        self.api_data.delete_created_folder()
        self.assertIn(
            "Directory 'output/cg_data' not found.",
            stdout.getvalue().strip(),
        )

    @patch("sys.stdout", new_callable=StringIO)
    @patch("data_ingestion.extract_student_data_from_API.shutil")
    def test_delete_created_folder_failure_permission_error(self, mock_shutil, stdout):
        """Test the file not permission error."""
        self.api_data.create_folder_path = "output/cg_data"
        mock_shutil.rmtree.side_effect = PermissionError()
        self.api_data.delete_created_folder()
        self.assertIn(
            "Permission denied to delete 'output/cg_data'.",
            stdout.getvalue().strip(),
        )

    @patch("sys.stdout", new_callable=StringIO)
    @patch("data_ingestion.extract_student_data_from_API.shutil")
    def test_delete_created_folder_failure_os_error(self, mock_shutil, stdout):
        """Test the file output error."""
        self.api_data.create_folder_path = "output/cg_data"
        mock_shutil.rmtree.side_effect = OSError()
        self.api_data.delete_created_folder()
        self.assertIn(
            "Error deleting directory:",
            stdout.getvalue().strip(),
        )

    @patch("data_ingestion.extract_student_data_from_API.load_dotenv")
    @patch("data_ingestion.extract_student_data_from_API.codegrade")
    @patch("data_ingestion.extract_student_data_from_API.os")
    def test_main(self, mock_os, mock_codegrade, mock_loadenv):
        """Test case: Test main() ."""
        mock_loadenv.return_value = None
        mock_codegrade.login.return_value = self.mock_client
        mock_os.getenv.return_value = None
        main()
        mock_codegrade.login.assert_called_once()


if __name__ == "__main__":
    unittest.main()
