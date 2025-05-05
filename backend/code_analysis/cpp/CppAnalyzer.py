"""
Created by Daniel Levy, 4/19/2025.

This script is designed to analyze C++ blacklisted keywords from
student submissions. The main central script will call this file and
turn a student's submission into an AST that will then be analyzed for
the use of unauthorized keywords. Once all submissions have been checked,
the data will then be sent back to the central script for output file
generation.
"""

import os

from clang import cindex
from clang.cindex import CursorKind


class CppAnalyzer:
    """Create object to perform keyword analysis."""

    __words = None
    __subDir = None
    __assignment = None
    __students: dict = None

    # Methods
    def __init__(self, words, subDir, assignment):
        """Construct CppAnalyzer object."""
        self.__words = words
        self.__subDir = subDir
        self.__assignment = assignment
        self.__students = dict(dict(dict()))

    """
        This is a helper method for runAnalysis. The goal is to parse each
        student's input file and manually verify whether or not they used
        blacklisted keywords and/or headers. All information will be written
        into the JSON file.
    """

    def generateAST(self):
        """Interface to the CppAnalyzer class."""
        for f in os.listdir(self.__subDir):
            if not f.endswith(".json"):
                studentName = f"{f.split('_')[2]} {f.split('_')[3]}"
                with open(f"{self.__subDir}/{f}/main.cpp", "r"):
                    headers = self.__checkHeaders(f"{self.__subDir}/{f}/main.cpp")

                    program = cindex.Index.create()
                    ast = program.parse(
                        f"{self.__subDir}/{f}/main.cpp",
                        args=["-std=c++11", "-nostdinc", "-nostdlibinc"],
                    )
                    self.__checkAST(ast.cursor, studentName)

                    if self.__students:
                        if headers:
                            if studentName not in self.__students:
                                self.__students[studentName] = {
                                    "totalFound": 0,
                                    "headers": list(),
                                    "wordsFound": dict(dict()),
                                }
                            self.__students[studentName]["headers"] = headers

        return self.__students

    """
        This method is designed to check all import statements at the start
        of a student's file and manually verify whether or not a library
        specified by the user was found inside a student's file. All we are
        going to do is add the library name so it can be saved in the JSON file
    """

    def __checkHeaders(self, file):
        foundHeaders = list()
        with open(file, "r") as iFile:
            includeFound = False
            for line in iFile:
                if line.startswith("#include"):
                    includeFound = True
                    headerName = ""
                    if line.find("<"):
                        headerName = line[line.find("<") + 1 : line.find(">")]
                    else:
                        headerName = line[line.find(" ") + 1 : line.find(" ")]

                    for w in self.__words:
                        if w == headerName:
                            foundHeaders.append(w)
                elif includeFound:
                    return foundHeaders

    """
        This method will analyze the AST for a student's input file and
        check if there any usages of keywords they weren't suppose to use.
        In this case, we are checking if the words used are not identifiers,
        so we manually verify if the current node represents a VARDECL or a
        NAMEEXPR first prior to adding the result to our error array. For right
        now, we note which word was used followed by the line number it came
        from.
    """

    def __checkAST(self, currNode, studentName):
        for i, w in enumerate(self.__words):
            if w == currNode.spelling:
                if currNode.kind == CursorKind.CALL_EXPR:
                    if currNode.location.file.name == "main.cpp":
                        continue

                if studentName not in self.__students:
                    self.__students[studentName] = {
                        "totalFound": 0,
                        "headers": list(),
                        "wordsFound": dict(dict()),
                    }
                if w not in self.__students[studentName]["wordsFound"]:
                    self.__students[studentName]["wordsFound"][w] = {
                        "count": 0,
                        "positions": list(),
                    }

                self.__students[studentName]["wordsFound"][w]["count"] += 1
                self.__students[studentName]["wordsFound"][w]["positions"].append(
                    f"<{currNode.location.line}.{currNode.location.column}>"
                )

        for c in currNode.get_children():
            self.__checkAST(c, studentName)
