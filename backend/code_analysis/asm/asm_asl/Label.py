"""
Created by Daniel Levy, 4/25/2025.

This is a class representing an x86 Label. This class
only contains the name of the label and that is it.
"""

from code_analysis.asm.asm_asl.ASL import ASL
from code_analysis.asm.asm_token.Token import Token


class Label(ASL):
    """Create object to represent Label."""

    __name: str = None

    def __init__(self, labelToken: Token):
        """Construct Label object."""
        super().__init__(labelToken.getStartPos(), labelToken.getEndPos())
        self.__name = labelToken.getLexeme()

    def toString(self):
        """Return label name."""
        return f"{self.__name}"
