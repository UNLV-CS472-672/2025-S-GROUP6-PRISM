"""
Created by Daniel Levy, 4/21/2025.

This script serves as the central script to perform the manual
checking of banned keywords inside of a student submission. This
class will primarily handle the I/O setup for the keyword analysis,
and it will interface with the correct script depending on the course
and programming language being used. At the end of the script, a JSON
file will be generated in this directory that can then be used by the
front end.
"""

import json
import os
import sys

from code_analysis.asm.AsmAnalyzer import AsmAnalyzer
from code_analysis.cpp.CppAnalyzer import CppAnalyzer


class KeywordAnalyzer:
    """Create object to perform keyword analysis."""

    __words: set = None
    __course: str = None
    __assignment: str = None
    __jsonFile = None
    __sections = None

    def __init__(self):
        """Construct KeywordAnalyzer object."""
        self.__sections = dict(dict())
        self.__openAndValidateFile()
        self.__createOutputFile()
        self.__runAnalysis()

    """
        This method is responsible for opening the input file
        containing the banned keywords, storing them into the
        KeywordAnalyzer's words array, and also getting the
        assignment for which the keyword analysis will be done for
    """

    def __openAndValidateFile(self):
        # ERROR CHECK #1: Make sure the front end passed in the appropriate
        #                 file arguments to this script
        if len(sys.argv) != 2:
            print("Usage: python3 KeywordAnalyzer.py <fileName>.txt")
            exit(1)
        elif not sys.argv[1].endswith(".txt"):
            print("Error! A .txt file was not passed as an argument")
            exit(1)

        # ERROR CHECK #2: Make sure the file exists in the
        #                 keyword_detection package
        if not os.path.exists(sys.argv[1]):
            print("Error! Banned keyword file does not exist!")
            exit(1)

        fileParts = sys.argv[1].split("_")
        self.__course = fileParts[0][2:]
        self.__assignment = fileParts[1][2:]

        with open(sys.argv[1], "r") as iFile:
            self.__words = set()
            for line in iFile.readlines():
                self.__words.add(line.strip("\n"))

    def __createOutputFile(self):
        self.__jsonFile = open(
            f"{self.__course}_{self.__assignment}_Found_Words.json", "w"
        )

    """
        This is the main method for KeywordAnalyzer. We will do a bulk analysis
        for every section by looking at student file located in each directory.
        Once this has been completed, we can then access a JSON file that will
        contain all the students who violated the rules for the assignment.
    """

    def __runAnalysis(self):
        submissionDir = (
            f"/PRISM/data/assignments/assignment_{self.__assignment}/bulk_submission"
        )

        for f in os.listdir(submissionDir):
            if not f.endswith(".csv"):
                fileParts = f.split("_")
                section = fileParts[2]
                students = dict()

                if self.__course == "135":
                    cpp = CppAnalyzer(
                        self.__words, f"{submissionDir}/{f}", self.__assignment
                    )
                    students = cpp.generateAST()
                elif self.__course == "218":
                    asm = AsmAnalyzer(
                        self.__words, f"{submissionDir}/{f}", self.__assignment
                    )
                    students = asm.tokenizeAssembly()

                self.__sections[section] = students
                for w in students:
                    totalCount = 0
                    for c in students[w]["wordsFound"]:
                        totalCount += students[w]["wordsFound"][c]["count"]
                    students[w]["totalFound"] = totalCount

        json.dump(self.__sections, self.__jsonFile, indent=4)


def main():
    """Interfaces with KeywordAnalyzer class."""
    KeywordAnalyzer()


if __name__ == "__main__":
    main()
