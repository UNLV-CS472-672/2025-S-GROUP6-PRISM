"""
Created by Daniel Levy, 4/25/2025.

This is a class representing an Instruction in x86.
An instruction object will store the instruction it
represents followed by any additional info the instruction
needs to perform its job correctly.
"""

from code_analysis.asm.asm_asl.ASL import ASL


class Instruction(ASL):
    """Create object to represent Instruction."""

    __instruction: str = None
    __extraInfo = None

    def __init__(self, startPos, endPos, instr, extra):
        """Construct Instruction object."""
        super().__init__(startPos, endPos)
        self.__instruction = instr
        self.__extraInfo = extra

    def toString(self):
        """Return instruction string."""
        return f"{self.__instruction}"
