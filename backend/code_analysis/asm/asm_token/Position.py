"""
Created by Daniel Levy, 4/21/2025.

This is a simple class designed to represent a position. A position
will be denoted by a line and a column, and we will use these positions
to keep track of where each token is located in relation to the file.
"""


class Position:
    """Create object to store position in file."""

    __line = None  # Line Placement
    __col = None  # Column Placement

    def __init__(self, startLine=0, startCol=0):
        """Construct Position object."""
        self.__line = startLine
        self.__col = startCol

    """ This is a getter method to retrieve the line. """

    def getLine(self):
        """Get line."""
        return self.__line

    """ This is a getter method to retrieve the column."""

    def getCol(self):
        """Get column."""
        return self.__col

    """
        This is a helper method to print out the current
        position for debugging purposes.
    """

    def toString(self):
        """Turn position to string."""
        return f"<{self.__line}.{self.__col}>"
