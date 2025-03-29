"""Created by Daniel Levy, 3/17/2025.

This is a factory class to generate a new Error object. This will be used for
the Error builder in case we have multiple errors while performing data
ingestion (which is unlikely).
"""

import data_ingestion.errors.DataIngestionError as error


class DataIngestionErrorFactory:
    """A factory class for creating instances of data ingestion errors.

    Methods:
        createError():
            Creates and returns an instance of DataIngestionError.
    """

    def createError(self):
        """Create and returns an instance of the DataIngestionError class.

        Returns:
            DataIngestionError: An instance of the DataIngestionError class.
        """
        return error.DataIngestionError()
