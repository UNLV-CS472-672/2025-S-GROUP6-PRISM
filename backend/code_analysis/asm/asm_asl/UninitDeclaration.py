"""
Created by Daniel Levy, 4/25/2025.

This is a class representing an uninit declaration in x86.
This type of declaration will only have a name, the size of
the data in bytes, and any values initialized if we are trying
to create an array of values.
"""

from code_analysis.asm.asm_asl.ASL import ASL


class UninitDeclaration(ASL):
    """Create object to represent an UninitDeclaration."""

    __name: str = None
    __size: str = None
    __values = None

    def __init__(self, startPos, endPos, name, size, values):
        """Construct UninitDeclaration object."""
        super().__init__(startPos, endPos)
        self.__name = name
        self.__size = size
        self.__values = values

    def toString(self):
        """Return name of uninit variable."""
        return f"{self.__name}"
