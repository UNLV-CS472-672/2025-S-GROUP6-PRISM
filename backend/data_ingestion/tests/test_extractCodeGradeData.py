"""Unit tests for `extractCodeGradeData.py` by Daniel Levy, 3/26/2025.

This test suite validates the behavior of the CodeGradeDataIngestion class.

It verifies:
- Successful processing of valid ZIP, JSON, and CSV files.
- Detection of various error conditions such as:
  - Duplicate ZIPs
  - Missing JSON or CSV files
  - Submission mismatches
  - Metadata inconsistencies
"""

import pytest
import json
import csv
import shutil
import os

from data_ingestion.extractCodeGradeData import CodeGradeDataIngestion


@pytest.mark.django_db
class TestExportCodeGradeData:
    """Test suite for CodeGradeDataIngestion error handling and data processing."""

    test_directory = "codegrade_data_test_directory"
    test_file = "CS 135 1001 - 2024 Fall - Assignment 0"
    errorFileName = "codegrade_data_errors.json"

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up and tear down the test environment for each test."""
        if not os.path.isdir(self.test_directory):
            os.mkdir(self.test_directory)

        self.ingest = CodeGradeDataIngestion(self.test_directory)

        self.createSubmissions(
            ["0 - Mary Smith", "23 - James Johnson", "34 - John Jones"]
        )

        self.createJSON(
            {
                "submission_ids": {
                    "0 - Mary Smith": 0,
                    "23 - James Johnson": 23,
                    "34 - John Jones": 34,
                },
                "user_ids": {
                    "0 - Mary Smith": 100000,
                    "23 - James Johnson": 100001,
                    "34 - John Jones": 100003,
                },
            }
        )

        self.createZIP(self.test_file)

        self.createCSV(
            [
                ["Id", "Username", "Name", "Grade"],
                ["100000", "smithm", "Mary Smith", "79.1"],
                ["100001", "johnsj", "James Johnson", "81.7"],
                ["100003", "jonesj", "John Jones", "63.2"],
            ],
            self.test_file,
        )

        yield

        shutil.rmtree(self.test_directory)
        if self.errorFileName in os.listdir():
            os.remove(self.errorFileName)

    def createSubmissions(self, zipData):
        """Create folders and placeholder files for simulated student submissions."""
        for z in zipData:
            path = os.path.join(self.test_directory, z)
            os.mkdir(path)
            open(os.path.join(path, "main.cpp"), "w").close()

    def createJSON(self, jsonData):
        """Write submission/user metadata to `.cg-info.json`."""
        with open(os.path.join(self.test_directory, ".cg-info.json"), "w") as jFile:
            json.dump(jsonData, jFile, indent=4)

    def createZIP(self, fileName):
        """Create ZIP archive of the test directory and move it back in."""
        shutil.make_archive(fileName, "zip", self.test_directory)
        shutil.rmtree(self.test_directory)
        os.mkdir(self.test_directory)
        shutil.move(
            f"{fileName}.zip", os.path.join(self.test_directory, f"{fileName}.zip")
        )

    def createCSV(self, csvData, fileName):
        """Create a CSV file in the test directory with given data."""
        with open(os.path.join(self.test_directory, f"{fileName}.csv"), "w") as file:
            csv.writer(file).writerows(csvData)

    def runAndProduceError(self):
        """Run extraction and return first error message from generated JSON."""
        self.ingest.extractData()
        assert self.errorFileName in os.listdir()

        with open(self.errorFileName, "r") as errors:
            jsonFile = json.load(errors)
            return jsonFile["errors"][0]["_DataIngestionError__msg"]

    def test_valid_data(self):
        """Ensure valid data does not generate an error file."""
        self.ingest.extractData()
        assert self.errorFileName not in os.listdir()

    def test_duplicate_zip_files_found(self):
        """Detect and report duplicate .zip files in the directory."""
        open(
            os.path.join(
                self.test_directory,
                f"{
                    self.test_file}.zip",
            ),
            "w",
        ).close()
        expected = f"A duplicate .zip file was found containing student submission in {
            self.test_directory}"
        assert self.runAndProduceError() == expected

    def test_missing_cg_json_file(self):
        """Detect missing `.cg-info.json` in zip contents."""
        self.createSubmissions(["0 - Mary Smith"])
        self.createZIP("CS 135 1001 - 2024 Fall - Assignment 1")
        assert self.runAndProduceError() == "The .cg-info.json file is missing."

    def test_invalid_csv_file_name(self):
        """Detect CSV file mismatch based on expected name."""
        self.createSubmissions(["0 - Mary Smith"])
        self.createJSON(
            {
                "submission_ids": {"0 - Mary Smith": 0},
                "user_ids": {"0 - Mary Smith": 100000},
            }
        )
        self.createZIP("CS 135 1001 - 2024 Fall - Assignment 1")
        self.createCSV(["error"], "CS 135 1001 - 2024 Fall - Assignment 2")
        expected = (
            "CS 135 1001 - 2024 Fall - Assignment 1.csv was not found in "
            f"{self.test_directory}."
        )
        assert self.runAndProduceError() == expected

    def test_non_matching_student_submission_id(self):
        """Detect when student folder ID does not match JSON submission ID."""
        self.createSubmissions(["0 - Mary Smith"])
        self.createJSON(
            {
                "submission_ids": {"0 - Mary Smith": 2},
                "user_ids": {"0 - Mary Smith": 100000},
            }
        )
        self.createZIP("CS 135 1001 - 2024 Fall - Assignment 1")
        self.createCSV(["error"], "CS 135 1001 - 2024 Fall - Assignment 1")
        expected = "The submission ID #0 for Mary Smith is not correct."
        assert self.runAndProduceError() == expected

    def test_no_student_metadata(self):
        """Detect absence of student metadata in CSV file."""
        self.createSubmissions(["0 - Mary Smith"])
        self.createJSON(
            {
                "submission_ids": {"0 - Mary Smith": 0},
                "user_ids": {"0 - Mary Smith": 100000},
            }
        )
        self.createZIP("CS 135 1001 - 2024 Fall - Assignment 2")
        self.createCSV(
            [["Id", "Username", "Name", "Grade"]],
            "CS 135 1001 - 2024 Fall - Assignment 2",
        )
        expected = "User ID 100000 does not have any metadata associated with it."
        assert self.runAndProduceError() == expected

    def test_multiple_student_metadata(self):
        """Detect duplicate user ID entries in the metadata file."""
        self.createSubmissions(["0 - Mary Smith"])
        self.createJSON(
            {
                "submission_ids": {"0 - Mary Smith": 0},
                "user_ids": {"0 - Mary Smith": 100000},
            }
        )
        self.createZIP("CS 135 1001 - 2024 Fall - Assignment 3")
        self.createCSV(
            [
                ["Id", "Username", "Name", "Grade"],
                ["100000", "smithm", "Mary Smith", "79.1"],
                ["100000", "smithm", "Mary Smith", "78.1"],
            ],
            "CS 135 1001 - 2024 Fall - Assignment 3",
        )
        expected = "User ID 100000 has multiple metadata entries associated with it."
        assert self.runAndProduceError() == expected

    def test_student_name_not_matching_user_id(self):
        """Detect name mismatch between metadata and user ID."""
        self.createSubmissions(["0 - Mary Smith"])
        self.createJSON(
            {
                "submission_ids": {"0 - Mary Smith": 0},
                "user_ids": {"0 - Mary Smith": 100000},
            }
        )
        self.createZIP("CS 135 1001 - 2024 Fall - Assignment 4")
        self.createCSV(
            [
                ["Id", "Username", "Name", "Grade"],
                ["100000", "smithm", "Mary Jane", "79.1"],
            ],
            "CS 135 1001 - 2024 Fall - Assignment 4",
        )
        expected = "User ID 100000 does not match the given name in the metadata file."
        assert self.runAndProduceError() == expected

    def test_student_missing_submission_in_zip(self):
        """Detect missing submission folder for a student defined in JSON."""
        self.createSubmissions(["23 - Paul Jones"])
        self.createJSON(
            {
                "submission_ids": {
                    "0 - Mary Smith": 0,
                    "23 - Paul Jones": 23,
                },
                "user_ids": {
                    "0 - Mary Smith": 100000,
                    "23 - Paul Jones": 100001,
                },
            }
        )
        self.createZIP("CS 135 1001 - 2024 Fall - Assignment 5")
        self.createCSV(
            [
                ["Id", "Username", "Name", "Grade"],
                ["100000", "smithm", "Mary Jane", "79.1"],
                ["100001", "jonesj", "Paul Jones", "89.3"],
            ],
            "CS 135 1001 - 2024 Fall - Assignment 5",
        )
        expected = "Submission for Mary Smith is missing in zip directory."
        assert self.runAndProduceError() == expected
