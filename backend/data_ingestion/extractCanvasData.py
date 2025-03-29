"""
Ingest Canvas metadata into the database.

Created by Daniel Levy, 3/17/2025.

We are primarily concerned with models from the `courses` app. We will also
manually validate that there are no errors in the provided Canvas gradebook files.
"""

# Django setup
import os
import django

import pandas as pd
from data_ingestion.errors.DataIngestionError import DataIngestionError
import data_ingestion.errors.DataIngestionErrorBuilder as eb
from courses.models import Semester, Class

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prism_backend.settings")
django.setup()


class CanvasDataIngestion:
    """
    Handle the ingestion and validation of Canvas metadata into the database.

    This class reads Canvas gradebook CSV files, validates their structure and
    metadata, and then populates the database with Semester and Course records.
    """

    # Fields
    __dirName = None  # Directory where Canvas data is located
    __fileName = None  # Current file we are checking
    __data = None  # DataFrame to store file contents
    __course = None
    __section = None
    __year = None
    __semester = None  # Spring, Summer, or Fall
    __metaID = None  # List of Canvas metadata info
    __courseID = None  # Course ID: Last number in the metaID
    __errors = None  # List of errors

    errors = list()  # Static list to keep track of all errors

    # Methods
    def __init__(self, dirName):
        """
        Initialize an instance of CanvasDataIngestion.

        Args:
            dirName (str): The directory where Canvas data is located.
        """
        self.__dirName = dirName
        self.__fileName = ""
        self.__data = ""
        self.__course = ""
        self.__section = ""
        self.__year = ""
        self.__semester = ""
        self.__metaID = list()
        self.__courseID = ""
        self.__errors = list()

    def __parseCanvasFileName(self, file):
        """
        Parse the Canvas file name to extract course, section, and semester info.

        Args:
            file (str): The file name of the CSV being processed.
        """
        fields = file.split("-")

        courseInfo = fields[3].split("_")
        self.__course = courseInfo[0] + courseInfo[1]
        self.__section = courseInfo[2]

        semesterInfo = fields[4].split("_")
        self.__year = semesterInfo[1]
        self.__semester = semesterInfo[2].removesuffix(".csv")
        self.__fileName = file

    def __convertToDataFrame(self):
        """
        Convert the contents of a CSV file into a Pandas DataFrame.

        Reads the CSV file located in __dirName and stores it in __data.
        """
        csvFile = open(f"{self.__dirName}/{self.__fileName}", "r")
        self.__data = pd.read_csv(csvFile)
        csvFile.close()

    def __validateData(self):
        """
        Validate the dataset by performing checks against the file name and meta ID.

        Checks include ensuring that SIS User ID matches SIS Login ID, verifying that
        the semester, course, and section values match the file name, and confirming
        that all rows share the same Canvas meta ID.
        """
        self.__metaID = self.__data["Section"].iloc[0]
        rowCount = 1
        for index, student in self.__data.iterrows():
            studentNameFields = student["Student"].split(",")
            studentName = studentNameFields[1][1:] + " " + studentNameFields[0]

            # Error Check #1: Make sure the student's User/login ID match
            if student["SIS User ID"] != student["SIS Login ID"]:
                self.__errors.append(
                    eb.DataIngestionErrorBuilder()
                    .addFileName(self.__fileName)
                    .addLine(rowCount)
                    .addMsg(
                        f"The User ID for {studentName} does not match the Login ID"
                    )
                    .createError()
                )

            self.courseInfo = self.__getCourseMetaData(student["Section"], rowCount)

            # Error Check #2: Make sure the semester matches
            if self.courseInfo[0] != self.__semester:
                self.__errors.append(
                    eb.DataIngestionErrorBuilder()
                    .addFileName(self.__fileName)
                    .addLine(rowCount)
                    .addMsg(
                        f"The semester for {studentName} does not match the Canvas semester."
                    )
                    .createError()
                )

            # Error Check #3: Make sure the course matches
            if self.courseInfo[1] != self.__course:
                self.__errors.append(
                    eb.DataIngestionErrorBuilder()
                    .addFileName(self.__fileName)
                    .addLine(rowCount)
                    .addMsg(
                        f"The course name for {studentName} does not match the Canvas course name."
                    )
                    .createError()
                )

            # Error Check #4: Make sure the section matches
            if self.courseInfo[2] != self.__section:
                self.__errors.append(
                    eb.DataIngestionErrorBuilder()
                    .addFileName(self.__fileName)
                    .addLine(rowCount)
                    .addMsg(
                        f"The section number for {studentName} does not match the Canvas section."
                    )
                    .createError()
                )

            # Error Check #5: Make sure the Canvas meta ID matches
            if student["Section"] != self.__metaID:
                self.__errors.append(
                    eb.DataIngestionErrorBuilder()
                    .addFileName(self.__fileName)
                    .addLine(rowCount)
                    .addMsg(f"The Canvas metadata ID does not match for {studentName}.")
                    .createError()
                )

            rowCount += 1

    def __getCourseMetaData(self, metaID, row):
        """
        Extract and process metadata for a course from the provided Canvas meta ID.

        Args:
            metaID (str): The Canvas meta ID from the CSV row.
            row (int): The current row number for error reporting.

        Returns:
            list: A list containing the derived semester, course, section, and course ID.
        """
        courseInfo = metaID.split("-")
        metaData = list()

        # The semester will be denoted by the last number in the first field
        if courseInfo[0][-1] == "2":
            metaData.append("Sprg")
        elif courseInfo[0][-1] == "5":
            metaData.append("Sumr")
        elif courseInfo[0][-1] == "8":
            metaData.append("Fall")
        else:
            self.__errors.append(
                eb.DataIngestionErrorBuilder()
                .addFileName(self.__fileName)
                .addLine(row)
                .addMsg("The semester is invalid.")
                .createError()
            )

        metaData.append(courseInfo[1] + courseInfo[2])
        metaData.append(courseInfo[3][3:])
        metaData.append(courseInfo[4])

        self.__courseID = courseInfo[4]
        return metaData

    def __populateDatabase(self):
        """
        Populate the database with Semester and Course information.

        This uses the validated data to create or retrieve Semester and
        Course entries in the database.
        """
        try:
            semester = Semester.objects.get_or_create(
                name=self.__semester + self.__year
            )
        except Semester.DoesNotExist:
            semester = Semester(name=self.__semester + self.__year)
            semester.save()

        try:
            className = Class.objects.get_or_create(name=self.__course)
        except Class.DoesNotExist:
            className = Class(name=self.__course)
            className.save()

    def extractData(self):
        """
        Extract and process data from files in the specified directory.

        Iterates over CSV files, validates them, and populates the database if
        there are no errors.
        """
        for file in os.listdir(self.__dirName):

            # Error Check #1: Verify each file is indeed a .csv file
            if not file.endswith(".csv"):
                self.errors.append(
                    eb.DataIngestionErrorBuilder()
                    .addFileName(file)
                    .addMsg(f"The file '{file}' is not a valid .csv file.")
                    .createError()
                )
                continue

            self.__parseCanvasFileName(file)
            self.__convertToDataFrame()
            self.__validateData()

            if len(self.__errors) > 0:
                for e in self.__errors:
                    self.errors.append(e)
                self.__errors = list()
                continue

            self.__populateDatabase()

        if len(CanvasDataIngestion.errors) > 0:
            DataIngestionError.createErrorJSON(
                "canvas_data_errors", CanvasDataIngestion.errors
            )
            CanvasDataIngestion.errors = list()


def main():
    """
    Run the Canvas data ingestion process.

    This function creates a CanvasDataIngestion instance and calls extractData.
    """
    ingest = CanvasDataIngestion("canvas_data")
    ingest.extractData()


if __name__ == "__main__":
    main()
