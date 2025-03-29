"""Unit tests for `extractCanvasData.py` by Daniel Levy, 3/26/2025.

This test suite validates the behavior of the CanvasDataIngestion class.

It verifies:
- Successful data extraction for valid input.
- Error detection and reporting for various edge cases (e.g., mismatches in
  section, semester, user ID, login ID, Canvas metadata ID, etc.).
"""

import pytest
import json
import csv
import os

from data_ingestion.extractCanvasData import CanvasDataIngestion


@pytest.mark.django_db
class TestExportCanvasData:
    """Test suite for CanvasDataIngestion functionality."""

    test_directory = "canvas_data_test_directory"
    fileName = f"{test_directory}/2025-03-25T0637_Grades-CS_135_1000_-_2024_Sumr.csv"
    errorFileName = "canvas_data_errors.json"

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up and tear down the test environment."""
        if not os.path.isdir(self.test_directory):
            os.mkdir(self.test_directory)

        self.ingest = CanvasDataIngestion(self.test_directory)

        dummy_data = [
            [
                "Student",
                "ID",
                "SIS User ID",
                "SIS Login ID",
                "Integration ID",
                "Section",
            ],
            [
                "Student, Test",
                "0000",
                "studet",
                "studet",
                "5000000000",
                "2245-CS-135-SEC1000-50000",
            ],
        ]

        with open(self.fileName, "w") as file:
            csv.writer(file).writerows(dummy_data)

        yield

        for f in os.listdir(self.test_directory):
            os.remove(f"{self.test_directory}/{f}")
        os.rmdir(self.test_directory)

        if self.errorFileName in os.listdir():
            os.remove(self.errorFileName)

    def runAndProduceError(self):
        """Run data extraction and return the first error message."""
        self.ingest.extractData()
        assert self.errorFileName in os.listdir()

        with open(self.errorFileName, "r") as errors:
            jsonFile = json.load(errors)
            return jsonFile["errors"][0]["_DataIngestionError__msg"]

    def test_valid_student_data(self):
        """Ensure valid student data does not produce an error file."""
        self.ingest.extractData()
        assert self.errorFileName not in os.listdir()

    def test_invalid_file_in_gradebook_directory(self):
        """Ensure non-CSV files raise a file-type error."""
        open(f"{self.test_directory}/bad_file.txt", "w").close()

        expected = "The file 'bad_file.txt' is not a valid .csv file."
        assert self.runAndProduceError() == expected

    def test_non_matching_user_login_ids(self):
        """Ensure mismatched User ID and Login ID triggers an error."""
        with open(self.fileName, "a") as file:
            csv.writer(file).writerows(
                [
                    [
                        "Student3, Test",
                        "0000",
                        "stude",
                        "studet",
                        "5000000000",
                        "2245-CS-135-SEC1000-50000",
                    ]
                ]
            )

        expected = "The User ID for Test Student3 does not match the Login ID"
        assert self.runAndProduceError() == expected

    def test_non_matching_semester_ids(self):
        """Ensure mismatched semester in Canvas meta ID triggers an error."""
        with open(self.fileName, "a") as file:
            csv.writer(file).writerows(
                [
                    [
                        "Student4, Test",
                        "0000",
                        "studet",
                        "studet",
                        "5000000000",
                        "2248-CS-135-SEC1000-50000",
                    ]
                ]
            )

        expected = "The semester for Test Student4 does not match the Canvas semester."
        assert self.runAndProduceError() == expected

    def test_non_matching_course_names(self):
        """Ensure mismatched course name triggers an error."""
        with open(self.fileName, "a") as file:
            csv.writer(file).writerows(
                [
                    [
                        "Student5, Test",
                        "0000",
                        "studet",
                        "studet",
                        "5000000000",
                        "2245-CS-202-SEC1000-50000",
                    ]
                ]
            )

        expected = (
            "The course name for Test Student5 does not match the Canvas course name."
        )
        assert self.runAndProduceError() == expected

    def test_non_matching_section_ids(self):
        """Ensure mismatched section ID triggers an error."""
        with open(self.fileName, "a") as file:
            csv.writer(file).writerows(
                [
                    [
                        "Student6, Test",
                        "0000",
                        "studet",
                        "studet",
                        "5000000000",
                        "2245-CS-135-SEC1001-50000",
                    ]
                ]
            )

        expected = (
            "The section number for Test Student6 does not match the Canvas section."
        )
        assert self.runAndProduceError() == expected

    def test_non_matching_canvas_ids(self):
        """Ensure mismatched Canvas metadata ID triggers an error."""
        with open(self.fileName, "a", newline="") as file:
            csv.writer(file).writerows(
                [
                    [
                        "Student7, Test",
                        "0000",
                        "studet",
                        "studet",
                        "5000000000",
                        "3245-CS-135-SEC1000-50000",
                    ]
                ]
            )

        expected = "The Canvas metadata ID does not match for Test Student7."
        assert self.runAndProduceError() == expected

    def test_invalid_semester_id(self):
        """Ensure invalid semester ID format triggers an error."""
        with open(self.fileName, "a") as file:
            csv.writer(file).writerows(
                [
                    [
                        "Student8, Test",
                        "0000",
                        "studet",
                        "studet",
                        "5000000000",
                        "2249-CS-135-SEC1000-50000",
                    ]
                ]
            )

        assert self.runAndProduceError() == "The semester is invalid."
