"""Created by Eli Rosales, 3/24/2025.

This is the test script for "extract_student_data_from_API.py"

Uses unittest.mock to mock the outputs you would get form the codegrade
API as we do not want to send requests for these tests to the actual
codegrade servers.

Verifys:
    - most getter funcitons
    - good and bad zip files
    - 1 big function (download submission, 3 more needed)
        - Need to test:
        - extract_all_assignments(),
        - extract_csv(),
        - delete_created_folder()
"""
import unittest
from unittest.mock import MagicMock, patch, mock_open
from data_ingestion.extract_student_data_from_API import API_Data
import io
import os
import zipfile
import httpx


class TestAPIData(unittest.TestCase):
    """Test class for 'extract student data from API'."""

    def setUp(self):
        """Set up client and class object."""
        self.mock_client = MagicMock()
        self.api_data = API_Data(self.mock_client)

    def create_mock_assignment(self, assignment_id, name, lock_date=None, deadline=None, max_grade=100):
        """Helper function create a mock assignment."""
        assignment = MagicMock()
        assignment.id = assignment_id
        assignment.name = name
        assignment.lock_date = lock_date
        assignment.deadline = deadline
        assignment.max_grade = max_grade
        return assignment

    def create_mock_submission(self, submission_id, user_id, user_name, user_username, grade, group_name=None):
        """Helper function that creates a mock submission with all needed attributes."""
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

    def create_mock_rubric(self, rubric_items):
        """Helper function create a rubric with given params."""
        rubric = MagicMock()
        rubric.selected = []
        rubric.rubrics = []
        for item in rubric_items:
            selected_item = MagicMock()
            selected_item.achieved_points = item['achieved_points']
            selected_item.points = item['points']
            selected_item.multiplier = item['multiplier']
            rubric_item = MagicMock()
            rubric_item.header = item['header']
            rubric.selected.append(selected_item)
            rubric.rubrics.append(rubric_item)
        return rubric

# start testing
    def test_handle_maybe(self):
        """Test that we can extract data from input."""
        maybe_mock = MagicMock()
        maybe_mock.try_extract.return_value = "extracted value"
        result = self.api_data.handle_maybe(maybe_mock)
        self.assertEqual(result, "extracted value")

    @patch('data_ingestion.extract_student_data_from_API.os')
    def test_mkdir_success(self, mock_os):
        """Test mkdirs returns true and only called once."""
        mock_os.makedirs.return_value = None
        result = self.api_data.mkdir("test_dir")
        self.assertTrue(result)
        mock_os.makedirs.assert_called_once_with("test_dir", exist_ok=True)

    @patch('data_ingestion.extract_student_data_from_API.os')
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_mkdir_failure(self, mock_stderr, mock_os):
        """Exception handling and returns an error."""
        mock_os.makedirs.side_effect = Exception("mkdir failed")
        result = self.api_data.mkdir("test_dir")
        self.assertFalse(result)
        self.assertIn("mkdir failed", mock_stderr.getvalue())

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
        expected_info = {"Course-ID": "course_id", "Name": "course_name", "Created-Date": "created_date"}
        self.assertEqual(course_info, expected_info)

    def test_get_all_submissions_success(self):
        """Test for retrieving all submissions from an assignment."""
        assignment = self.create_mock_assignment("1234", "Test Assignment")
        self.mock_client.assignment = assignment
        self.mock_client.assignment.get_all_submissions.return_value = ["submission1", "submission2"]
        submissions = self.api_data.get_all_submissions(assignment)
        self.assertEqual(submissions, ["submission1", "submission2"])
        self.mock_client.assignment.get_all_submissions.assert_called_once_with(assignment_id='1234')

    def test_get_all_submissions_fail(self):
        """We get an exception from api call get_all_submissions."""
        assignment = self.create_mock_assignment("1234", "Test Assignment")
        self.mock_client.assignment = assignment
        self.mock_client.assignment.get_all_submissions.side_effect = Exception("invalid Assignment input")
        result = self.api_data.get_all_submissions(assignment)
        self.assertFalse(result)

    def test_get_all_graders(self):
        """Test to get all graders from assignment."""
        assignment = self.create_mock_assignment("assignment1", "Test Assignment")
        self.mock_client.assignment.get_all_graders.return_value = ["grader1", "grader2"]
        graders = self.api_data.get_all_graders(assignment)
        self.assertEqual(graders, ["grader1", "grader2"])
        self.mock_client.assignment.get_all_graders.assert_called_once_with(assignment_id="assignment1")

    def test_get_rubric(self):
        """Test rubric from api call."""
        assignment = self.create_mock_assignment("1234", "Test Assignment")
        mock_rubric_data = {"rubric_item": "value"}
        self.mock_client.assignment.get_rubric.return_value = mock_rubric_data
        rubric = self.api_data.get_rubric(assignment)
        self.assertEqual(rubric, mock_rubric_data)
        self.mock_client.assignment.get_rubric.assert_called_with(assignment_id="1234")

    def test_get_rubric_grade(self):
        """Test formatting of get rubric grade method."""
        mock_grade_data = {"grade_item": "value"}
        self.mock_client.submission.get_rubric_result.return_value = mock_grade_data
        grade = self.api_data.get_rubric_grade("sub123")
        self.assertEqual(grade, mock_grade_data)
        self.mock_client.submission.get_rubric_result.assert_called_with(submission_id="sub123")

    def test_get_rubric_grade_none(self):
        """Test the error handleing for a submission wo a rubric."""
        self.mock_client.submission.get_rubric_result.side_effect = Exception("Rubric not found")
        grade = self.api_data.get_rubric_grade("sub123")
        self.assertIsNone(grade)
        self.mock_client.submission.get_rubric_result.assert_called_with(submission_id="sub123")

    @patch('data_ingestion.extract_student_data_from_API.os')
    @patch('data_ingestion.extract_student_data_from_API.json')
    def test_get_json_file_success(self, mock_json, mock_os):
        """Test json is called and we create the output file .cg-info.json."""
        m = mock_open()
        with patch('builtins.open', m):
            mock_os.path.join.return_value = "test_path/.cg-info.json"
            mock_os.makedirs.return_value = None
            test_dict = {"key": "value"}
            self.api_data.get_json_file(test_dict, "test_path")
            mock_json.dump.assert_called()
            mock_os.makedirs.assert_called_once_with("test_path", exist_ok=True)

    @patch('data_ingestion.extract_student_data_from_API.os')
    @patch('data_ingestion.extract_student_data_from_API.json')
    def test_get_json_file_mkdir_failure(self, mock_json, mock_os):
        """Exception handling."""
        mock_os.makedirs.side_effect = Exception("mkdir failed")
        test_dict = {"key": "value"}
        self.api_data.get_json_file(test_dict, "test_path")
        mock_os.makedirs.assert_called_once_with("test_path", exist_ok=True)
        mock_json.dump.assert_not_called()

    @patch('data_ingestion.extract_student_data_from_API.codegrade')
    @patch('data_ingestion.extract_student_data_from_API.os')
    @patch('data_ingestion.extract_student_data_from_API.zipfile')
    @patch('data_ingestion.extract_student_data_from_API.io')
    def test_download_submission_success_individual(self, mock_io, mock_zipfile, mock_os, mock_codegrade):
        """Test that we can doanload a submission and that output file was created."""
        mock_submission = self.create_mock_submission("sub1", "user1", "User One", "user1", 90)
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
        with patch('builtins.open', mock_target_file):
            self.api_data.download_submission(mock_submission, "output_dir")
        mock_os.makedirs.assert_called_once_with(os.path.join("output_dir", "User One"), exist_ok=True)

    @patch('data_ingestion.extract_student_data_from_API.codegrade')
    @patch('data_ingestion.extract_student_data_from_API.os')
    @patch('data_ingestion.extract_student_data_from_API.zipfile')
    @patch('data_ingestion.extract_student_data_from_API.io')
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_download_submission_bad_zip(self, mock_stderr, mock_io, mock_zipfile, mock_os, mock_codegrade):
        """Test that the zipfile spits an error."""
        mock_submission = self.create_mock_submission("sub1", "user1", "User One", "user1", 90)
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
        mock_submission = self.create_mock_submission("sub1", "user1", "User One", "user1", 90)
        self.mock_client.file.download.side_effect = httpx.ReadError("Read error")
        with self.assertRaises(httpx.ReadError):
            self.api_data.download_submission(mock_submission, "output_dir")


if __name__ == '__main__':
    unittest.main()
