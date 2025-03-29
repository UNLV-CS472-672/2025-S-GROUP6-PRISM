"""
Created by Daniel Levy, 3/17/2025.

Error handling for PRISM data ingestion.

This module provides a class for capturing, managing, and exporting data
ingestion errors. Although the likelihood of errors is low, a safety net is
provided in case something unexpected happens.
"""

import json


class DataIngestionError:
    """Represents a single data ingestion error.

    This class is designed to store metadata about an error that occurred
    during the ingestion process, and to serialize that error to JSON.

    Attributes:
        __fileName (str): The name of the file where the error occurred.
        __line (str): The line in the file where the error occurred.
        __msg (str): A message describing the error.
    """

    __fileName = None
    __line = None
    __msg = None

    def __init__(self):
        """Initialize a new, empty DataIngestionError instance."""
        self.__fileName = ""
        self.__line = ""
        self.__msg = ""

    @staticmethod
    def createErrorJSON(fileName, errors):
        """Create a JSON file from a list of DataIngestionError objects.

        Args:
            fileName (str): The base name of the output file (no extension).
            errors (list): A list of DataIngestionError instances.

        Returns:
            None. Writes the errors to a file named "<fileName>.json".
        """
        errorCount = len(errors)
        with open(f"{fileName}.json", "w") as oFile:
            oFile.write('{\n\t"errors": [\n')
            for i, e in enumerate(errors):
                json.dump(e.__dict__, oFile, indent=4)
                if i != errorCount - 1:
                    oFile.write(",\n")
            oFile.write("\t]\n")
            oFile.write("}\n")

    def setFileName(self, fileName):
        """Set the file name where the error occurred.

        Args:
            fileName (str): The name of the file.
        """
        self.__fileName = fileName

    def setLine(self, line):
        """Set the line number where the error occurred.

        Args:
            line (str): The line number or line content.
        """
        self.__line = line

    def setMsg(self, msg):
        """Set the error message.

        Args:
            msg (str): A descriptive message about the error.
        """
        self.__msg = msg

    def getFileName(self):
        """Return the name of the file where the error occurred.

        Returns:
            str: The file name.
        """
        return self.__fileName

    def getLine(self):
        """Return the line number or content associated with the error.

        Returns:
            str: The line reference.
        """
        return self.__line

    def getMsg(self):
        """Return the error message.

        Returns:
            str: A message describing the error.
        """
        return self.__msg
