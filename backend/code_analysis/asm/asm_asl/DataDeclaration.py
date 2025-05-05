"""
Created by Daniel Levy, 4/25/2025.

This is a class representing an x86 data declaration. This
class contains the name of the variable, its size in terms
of bytes, a list of its values, and if the data declaration
represents a constant or not.
"""

from code_analysis.asm.asm_asl.ASL import ASL


class DataDeclaration(ASL):
    """Create object to represent Data Declaration."""

    __name: str = None
    __size: str = None
    __value = None

    __isConstant: bool = None

    def __init__(self, startPos, endPos, name, size, value, constant):
        """Construct DataDeclaration object."""
        super().__init__(startPos, endPos)
        self.__name = name
        self.__size = size
        self.__value = value
        self.__isConstant = constant

    def toString(self):
        """Return declaration name."""
        return f"{self.__name}"
