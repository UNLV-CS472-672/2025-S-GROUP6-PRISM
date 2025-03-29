"""
Created by Daniel Levy, 2/21/2025.

This script is responsible for the data ingestion of CodeGrade metadata
into the database. We are primarily concerned with models from the
`assignments` app. We will also manually validate that there are no
errors in the provided CodeGrade data files.
"""

import os
import json
import math
from zipfile import ZipFile

import django
import pandas as pd

from assignments.models import Student
import data_ingestion.errors.DataIngestionErrorBuilder as eb
from data_ingestion.errors.DataIngestionError import DataIngestionError

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prism_backend.settings")
django.setup()


class CodeGradeDataIngestion:
    """
    Class to ingest CodeGrade data.

    This class handles the ingestion of CodeGrade data, including the extraction
    of student submissions, validation of metadata, and population of the database.
    It performs various checks to ensure the integrity and completeness of the data.

    Attributes:
        __dirName (str): Directory containing all data (should be 'codegrade_data').
        __submissionFileName (str): Current course/assignment being processed.
        __className (str): Name of the class extracted from the ZIP file.
        __section (str): Section of the class extracted from the ZIP file.
        __semester (str): Semester information extracted from the ZIP file.
        __assignmentName (str): Assignment name extracted from the ZIP file.
        __zipFileDirectory (str): Directory containing unzipped student submissions.
        __submissions (list): List of CodeGrade submission IDs.
        __users (list): List of CodeGrade user IDs.
        __metaData (DataFrame): DataFrame containing CodeGrade metadata.
        __errors (list): List of errors encountered during data ingestion.
        fileSeen (set): Static set tracking all processed ZIP files.
        allErrors (list): Static list tracking all errors encountered.
    """

    __dirName = None
    __submissionFileName = None  # Current course/assignment being processed
    __className = None
    __section = None
    __semester = None
    __assignmentName = None
    __zipFileDirectory = None  # Directory containing unzipped student submissions
    __submissions = None  # List of CodeGrade submission IDs
    __users = None  # List of CodeGrade user IDs
    __metaData = None  # DataFrame containing CodeGrade metadata
    __errors = None

    fileSeen = set()  # Static set tracking processed ZIP files
    allErrors = list()  # Static list tracking all errors encountered

    def __init__(self, dirName):
        """
        Initialize the CodeGradeDataIngestion instance.

        Args:
            dirName (str): Name of the directory to be used for data processing.
        """
        self.__dirName = dirName
        self.__submissionFileName = ""
        self.__className = ""
        self.__section = ""
        self.__semester = ""
        self.__assignmentName = ""
        self.__zipFileDirectory = ""
        self.__submissions = list()
        self.__users = list()
        self.__metaData = list()
        self.__errors = list()

    def __extractStudentFilesFromZIP(self):
        """
        Extract student submission files from ZIP archives.

        Iterates through all files in the directory specified by __dirName.
        Identifies ZIP files that have not been processed (not in fileSeen),
        parses their names, and extracts their contents into a designated subdirectory.

        Raises:
            ValueError: If no new ZIP files are found or if duplicate ZIP files are detected.

        Side Effects:
            - Extracts ZIP file contents into a subdirectory.
            - Updates the fileSeen set.
            - Appends an error to __errors if an issue is encountered.
        """
        for file in os.listdir(self.__dirName):
            if file.endswith(".zip") and file not in CodeGradeDataIngestion.fileSeen:
                self.__parseZipFileName(file)
                self.__zipFileDirectory = f"{
                    self.__dirName}/{
                    self.__submissionFileName}"

                zipFile = ZipFile(f"{self.__dirName}/{file}")
                zipFile.extractall(self.__zipFileDirectory)

                CodeGradeDataIngestion.fileSeen.add(file)
                return

        self.__errors.append(
            eb.DataIngestionErrorBuilder()
            .addFileName(self.__dirName)
            .addMsg(
                f"A duplicate .zip file was found containing student submission in {
                    self.__dirName}"
            )
            .createError()
        )
        raise ValueError()

    def __parseZipFileName(self, name):
        """
        Parse the given ZIP file name to extract class-related attributes.

        Expected format: "<Class Name> <Section>-<Semester>-<Assignment Name>.zip"

        Sets the following attributes:
            __submissionFileName: Name without the ".zip" extension.
            __className: Class name extracted.
            __section: Section of the class.
            __semester: Semester information.
            __assignmentName: Assignment name.

        Args:
            name (str): The name of the ZIP file.
        """
        self.__submissionFileName = name.removesuffix(".zip")
        zipFields = name.split("-", 2)

        canvasName = zipFields[0].split(" ")
        self.__className = canvasName[0] + " " + canvasName[1]
        self.__section = canvasName[2]

        self.__semester = zipFields[1].strip()
        self.__assignmentName = zipFields[2][:-4].strip()

    def __checkIfJSONFileExists(self):
        """
        Check if the .cg-info.json file exists in the extracted ZIP directory.

        Appends an error if the file is missing.

        Raises:
            DataIngestionError: If the .cg-info.json file is not found.
        """
        if not os.path.exists(f"{self.__zipFileDirectory}/.cg-info.json"):
            self.__errors.append(
                eb.DataIngestionErrorBuilder()
                .addFileName(self.__dirName)
                .addMsg("The .cg-info.json file is missing.")
                .createError()
            )

    def __extractJSON(self):
        """
        Extract and process JSON data from the CodeGrade information file.

        Loads the .cg-info.json file and extracts submission and user IDs.

        Raises:
            FileNotFoundError: If the .cg-info.json file is missing.
            JSONDecodeError: If the JSON is malformed.
        """
        self.__checkIfJSONFileExists()

        with open(f"{self.__zipFileDirectory}/.cg-info.json", "r") as cgInfo:
            jsonStudentData = json.load(cgInfo)

        self.__submissions, self.__users = (
            jsonStudentData["submission_ids"],
            jsonStudentData["user_ids"],
        )

    def __extractMetaDataFromCSV(self):
        """
        Extract metadata from a CSV file in the directory.

        Searches for a CSV file matching __submissionFileName in __dirName.
        If found, reads it into a pandas DataFrame and stores it in __metaData.

        Raises:
            ValueError: If the CSV file is not found.

        Side Effects:
            Updates __metaData or logs an error.
        """
        for file in os.listdir(self.__dirName):
            if file == f"{self.__submissionFileName}.csv":
                with open(f"{self.__dirName}/{file}", "r") as csvFile:
                    df = pd.read_csv(csvFile)
                self.__metaData = df
                return

        self.__errors.append(
            eb.DataIngestionErrorBuilder()
            .addFileName(self.__submissionFileName)
            .addMsg(
                f"{
                    self.__submissionFileName}.csv was not found in {
                    self.__dirName}."
            )
            .createError()
        )
        raise ValueError()

    def __verifyStudentSubmissionExists(self):
        """
        Verify the existence and correctness of student submissions.

        Iterates through submission IDs and checks if the corresponding student file
        exists. Validates that the submission ID in the file matches the expected value.
        Logs an error if discrepancies are found.

        Raises:
            Exception: Catches exceptions during file existence check and continues.
        """
        for key, value in self.__submissions.items():
            try:
                subID, studentName = self.__checkIfStudentFileExists(key)
            except Exception:
                continue
            else:
                if subID != value:
                    self.__errors.append(
                        eb.DataIngestionErrorBuilder()
                        .addFileName(self.__zipFileDirectory)
                        .addMsg(
                            f"The submission ID #{subID} for {studentName} is not correct."
                        )
                        .createError()
                    )

    def __verifyStudentUserExistsInMetaData(self):
        """
        Verify that each student user has valid metadata.

        For each user, checks that:
          1. The user has a valid submission (exists in the ZIP directory).
          2. There is exactly one metadata entry.
          3. The student name matches the metadata.

        Raises:
            ValueError: If any check fails.
        """
        for key, value in self.__users.items():
            try:
                subID, studentName = self.__checkIfStudentFileExists(key)
            except Exception:
                raise ValueError()
            else:
                entry = self.__metaData.loc[self.__metaData["Id"] == value]
                entriesFound = len(entry)

                if entriesFound < 1:
                    self.__errors.append(
                        eb.DataIngestionErrorBuilder()
                        .addFileName(self.__submissionFileName)
                        .addMsg(
                            f"User ID {value} does not have any metadata associated with it."
                        )
                        .createError()
                    )
                    raise ValueError()
                elif entriesFound > 1:
                    self.__errors.append(
                        eb.DataIngestionErrorBuilder()
                        .addFileName(self.__submissionFileName)
                        .addMsg(
                            f"User ID {value} has multiple metadata entries associated with it."
                        )
                        .createError()
                    )
                    raise ValueError()

                if entry.iloc[0, 2] != studentName:
                    self.__errors.append(
                        eb.DataIngestionErrorBuilder()
                        .addFileName(self.__submissionFileName)
                        .addMsg(
                            f"User ID {value} does not match the given name in the metadata file."
                        )
                        .createError()
                    )
                    raise ValueError()

    def __checkIfStudentFileExists(self, fileName):
        """
        Check if a student's submission file exists in the ZIP directory.

        Args:
            fileName (str): Name of the student's submission file in the format "<submissionID>-<studentName>".

        Returns:
            tuple: (subID (int), studentName (str))

        Raises:
            ValueError: If the student's submission file is not found.
        """
        subID, studentName = fileName.split("-", 1)
        studentName = studentName.strip()
        subID = int(subID.strip())

        if fileName not in os.listdir(self.__zipFileDirectory):
            self.__errors.append(
                eb.DataIngestionErrorBuilder()
                .addFileName(self.__submissionFileName)
                .addMsg(f"Submission for {studentName} is missing in zip directory.")
                .createError()
            )
            raise ValueError()

        return subID, studentName

    def __populateDatabase(self):
        """
        Populate the database with student information from metadata.

        Iterates through the metadata DataFrame and uses the information to
        create or retrieve Student objects. The student's full name is split into
        first and last names. Email is constructed by appending "@unlv.nevada.edu" to the username.
        """
        for _, student in self.__metaData.iterrows():
            names = student["Name"].split(" ")

            try:
                currStudent = Student.objects.get_or_create(
                    email=student["Username"] + "@unlv.nevada.edu",
                    codeGrade_id=student["Id"],
                    username=student["Username"],
                    first_name=names[0],
                    last_name=names[1],
                )
            except Student.DoesNotExist:
                currStudent = Student(
                    email=student["Username"] + "@unlv.nevada.edu",
                    codeGrade_id=student["Id"],
                    username=student["Username"],
                    first_name=names[0],
                    last_name=names[1],
                )
                currStudent.save()

    def extractData(self):
        """
        Extract and process student submission data from the specified directory.

        Iterates through the submissions and performs the following steps:
          1. Extract student files from ZIP archives.
          2. Extract JSON data.
          3. Extract metadata from CSV files.
          4. Verify student submissions exist.
          5. Verify student users exist in the metadata.

        If errors occur during processing, they are collected and added to a global error list.
        After processing all submissions, any accumulated errors are written to a JSON file.
        If no errors occur, the database is populated with valid submission data.

        Raises:
            Exceptions are caught internally. Errors are logged and stored in a JSON file.

        Side Effects:
            - Updates global error list (allErrors).
            - Writes errors to a JSON file if any are encountered.
            - Populates the database with valid data.
        """
        submissionDirLength = len(os.listdir(self.__dirName))

        for i in range(math.ceil(submissionDirLength / 2)):
            try:
                self.__extractStudentFilesFromZIP()
                self.__extractJSON()
                self.__extractMetaDataFromCSV()
                self.__verifyStudentSubmissionExists()
                self.__verifyStudentUserExistsInMetaData()
            except Exception:
                for e in self.__errors:
                    CodeGradeDataIngestion.allErrors.append(e)
                self.__errors = list()
                continue
            else:
                self.__populateDatabase()

        if len(CodeGradeDataIngestion.allErrors) > 0:
            DataIngestionError.createErrorJSON(
                "codegrade_data_errors", CodeGradeDataIngestion.allErrors
            )
            CodeGradeDataIngestion.allErrors = list()
            CodeGradeDataIngestion.fileSeen = set()


def main():
    """
    Initiate the CodeGrade data ingestion process.

    Creates an instance of CodeGradeDataIngestion with the specified data source and
    calls its extractData method.

    Raises:
        Exceptions from instantiation or execution of extractData will propagate.
    """
    cgData = CodeGradeDataIngestion("codegrade_data")
    cgData.extractData()


if __name__ == "__main__":
    main()
