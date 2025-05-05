"""
Created by Daniel Levy, 4/25/2025.

This is an ASL (Abstract Syntax List) class and is suppose to
represent a list of data declarations and instructions for an
x86 input file. Currently, an ASL will consist of a starting
and ending position to denote where each line of code was found
in the x86 assembly file.
"""

from code_analysis.asm.asm_token.Position import Position


class ASL:
    """Create object to represent an ASL."""

    __startPos: Position = None
    __endPos: Position = None

    def __init__(self, start, end):
        """Construct ASL object."""
        self.__startPos = start
        self.__endPos = end

    def getStartPos(self):
        """Get start position for user."""
        return self.__startPos

    def getEndPos(self):
        """Get end position for user."""
        return self.__endPos
