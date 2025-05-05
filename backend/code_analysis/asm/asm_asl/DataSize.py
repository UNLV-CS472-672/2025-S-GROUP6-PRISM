"""
Created by Daniel Levy, 4/25/2025.

This is a class representing an x86 data size. Currently,
it is storing the size of the data followed by a list
that contains the dereferencing expression.
"""

from code_analysis.asm.asm_asl.ASL import ASL


class DataSize(ASL):
    """Create object to represent DataSize."""

    __size: str = None
    __lst = None

    def __init__(self, startPos, endPos, size, lst):
        """Construct DataSize object."""
        super().__init__(startPos, endPos)
        self.__size = size
        self.__lst = lst

    def toString(self):
        """Return data size string."""
        return f"{self.__size}"
