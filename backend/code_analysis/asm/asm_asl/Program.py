"""
Created by Daniel Levy, 4/25/2025.

This is a class representing an x86 Program. This class
contains all data declarations, uninit declarations, and
all Instructions associated with the x86 program we are
trying to parse.
"""

from code_analysis.asm.asm_asl.ASL import ASL


class Program(ASL):
    """Create object to represent x86 Program."""

    __init_data = None
    __uninit_data = None
    __code = None

    def __init__(self, startPos, endPos, init, uninit, code):
        """Construct Program object."""
        super().__init__(startPos, endPos)
        self.__init_data = init
        self.__uninit_data = uninit
        self.__code = code

    def toString(self):
        """Print each declaration and instruction for debugging."""
        print("Data Declarations: ")
        for v in self.__init_data:
            print(v.toString())

        print("\nUninit Declarations: ")
        for v in self.__uninit_data:
            print(v.toString())

        print("\nInstructions: ")
        for v in self.__code:
            print(v.toString())
