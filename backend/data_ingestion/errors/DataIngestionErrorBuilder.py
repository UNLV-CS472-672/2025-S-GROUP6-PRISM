"""
Created by Daniel Levy, 3/17/2025.

Error builder for PRISM data ingestion.

This is a simple error builder class that can be used for PRISM's data ingestion.
Every builder will create a new error, and we can add the error information we need
based on what we're checking.
"""

import data_ingestion.errors.DataIngestionErrorFactory as ef


class DataIngestionErrorBuilder:
    """Builder for constructing DataIngestionError objects.

    This class allows for building a DataIngestionError step by step
    using a fluent interface.

    Attributes:
        __error (DataIngestionError): The error object being built.
    """

    __error = None

    def __init__(self):
        """Initialize the builder with a new DataIngestionError instance."""
        self.__error = ef.DataIngestionErrorFactory().createError()

    def createError(self):
        """Return the constructed DataIngestionError object.

        Returns:
            DataIngestionError: The completed error object.
        """
        return self.__error

    def addFileName(self, fileName):
        """Set the file name for the error.

        Args:
            fileName (str): The name of the file where the error occurred.

        Returns:
            DataIngestionErrorBuilder: The builder instance for method chaining.
        """
        self.__error.setFileName(fileName)
        return self

    def addLine(self, line):
        """Set the line reference for the error.

        Args:
            line (int or str): The line number or content associated with the error.

        Returns:
            DataIngestionErrorBuilder: The builder instance for method chaining.
        """
        self.__error.setLine(line)
        return self

    def addMsg(self, msg):
        """Set the error message.

        Args:
            msg (str): A descriptive error message.

        Returns:
            DataIngestionErrorBuilder: The builder instance for method chaining.
        """
        self.__error.setMsg(msg)
        return self
