"""
Created by Daniel Levy, 4/25/2025.

This is a parser designed to parse an x86 file and return
an ASL. The purpose of this file is to allow for more advanced
assembly code analysis in the future since we are limited by tokens
which makes it harder to analyze code. All submissions in the database
should be valid, so this file would not produce any errors. If there is
any incomplete submission in the directory, then this class should be ignored.
"""

from code_analysis.asm.asm_asl.DataDeclaration import DataDeclaration
from code_analysis.asm.asm_asl.DataSize import DataSize
from code_analysis.asm.asm_asl.Instruction import Instruction
from code_analysis.asm.asm_asl.Label import Label
from code_analysis.asm.asm_asl.Program import Program
from code_analysis.asm.asm_asl.UninitDeclaration import UninitDeclaration
from code_analysis.asm.asm_token.Token import Token
from code_analysis.asm.asm_token.TokenType import TokenType


class Parser:
    """Create object to represent Parser."""

    __tokens = None
    __currPos = None
    __lookahead = None

    def __init__(self, tokens):
        """Construct Parser object."""
        self.__tokens = tokens
        self.__currPos = 0
        self.__lookahead: Token = tokens[0]

    def __consume(self):
        print(self.__lookahead.toString())
        self.__currPos += 1
        self.__lookahead = self.__tokens[self.__currPos]

    def __match(self, expected: TokenType):
        if self.__lookahead.getType() == expected:
            self.__consume()
            return True
        else:
            return False

    def __peek(self, expected: TokenType):
        return self.__lookahead.getType() == expected

    def __peekNext(self, expected: TokenType):
        if self.__currPos < len(self.__tokens) - 1:
            return self.__tokens[self.__currPos + 1].getType() == expected
        else:
            return False

    def __isRegister(self):
        return self.__lookahead.getType() in [
            TokenType.RAX,
            TokenType.EAX,
            TokenType.AX,
            TokenType.AL,
            TokenType.RBX,
            TokenType.EBX,
            TokenType.BX,
            TokenType.BL,
            TokenType.RCX,
            TokenType.ECX,
            TokenType.CX,
            TokenType.CL,
            TokenType.RDX,
            TokenType.EDX,
            TokenType.DX,
            TokenType.DL,
            TokenType.RSI,
            TokenType.ESI,
            TokenType.SI,
            TokenType.SIL,
            TokenType.RDI,
            TokenType.EDI,
            TokenType.DI,
            TokenType.DIL,
            TokenType.RBP,
            TokenType.EBP,
            TokenType.BP,
            TokenType.BPL,
            TokenType.RSP,
            TokenType.ESP,
            TokenType.SP,
            TokenType.SPL,
            TokenType.R8,
            TokenType.R8D,
            TokenType.R8W,
            TokenType.R8B,
            TokenType.R9,
            TokenType.R9D,
            TokenType.R9W,
            TokenType.R9B,
            TokenType.R10,
            TokenType.R10D,
            TokenType.R10W,
            TokenType.R10B,
            TokenType.R11,
            TokenType.R11D,
            TokenType.R11W,
            TokenType.R11B,
            TokenType.R12,
            TokenType.R12D,
            TokenType.R12W,
            TokenType.R12B,
            TokenType.R13,
            TokenType.R13D,
            TokenType.R13W,
            TokenType.R13B,
            TokenType.R14,
            TokenType.R14D,
            TokenType.R14W,
            TokenType.R14B,
            TokenType.R15,
            TokenType.R15D,
            TokenType.R15W,
            TokenType.R15B,
            TokenType.XMM0,
            TokenType.XMM1,
            TokenType.XMM2,
            TokenType.XMM3,
            TokenType.XMM4,
            TokenType.XMM5,
            TokenType.XMM6,
            TokenType.XMM7,
            TokenType.XMM8,
            TokenType.XMM9,
            TokenType.XMM10,
            TokenType.XMM11,
            TokenType.XMM12,
            TokenType.XMM13,
            TokenType.XMM14,
            TokenType.XMM15,
        ]

    # 1. <program> := <data>? <bss>? <text> ;
    def program(self):
        """Create ASL based on the tokens provided."""
        startPos = self.__lookahead.getStartPos()
        dataLst = uninitLst = text = None
        if self.__peek(TokenType.SECTION) and self.__peekNext(TokenType.ID_DATA):
            dataLst = self.__data()
        if self.__peek(TokenType.SECTION) and self.__peekNext(TokenType.ID_BSS):
            uninitLst = self.__bss()
        text = self.__text()

        program = Program(
            startPos, self.__lookahead.getEndPos(), dataLst, uninitLst, text
        )
        program.toString()

    # 2. <data> := 'section' '.data' <decl>* ;
    def __data(self):
        initList = list()
        self.__match(TokenType.SECTION)
        self.__match(TokenType.ID_DATA)
        while self.__peek(TokenType.ID):
            initList.append(self.__decl())

        return initList

    # 3. <decl> := <ID> (('equ' | 'db' | 'dw' | 'dd' | 'dq') <number> (',' <number>)*)* ;
    def __decl(self):
        name = self.__lookahead.getLexeme()
        startPos = self.__lookahead.getStartPos()
        isConstant = False
        self.__match(TokenType.ID)
        if self.__peek(TokenType.EQU):
            self.__match(TokenType.EQU)
            value = self.__lookahead.getLexeme()
            endPos = self.__lookahead.getEndPos()
            self.__match(TokenType.NUM)
            isConstant = True
            return DataDeclaration(startPos, endPos, name, None, value, isConstant)
        else:
            value = list()
            size = None
            endPos = None
            while (
                self.__peek(TokenType.DB)
                or self.__peek(TokenType.DW)
                or self.__peek(TokenType.DD)
                or self.__peek(TokenType.DQ)
            ):
                size = self.__lookahead.getLexeme()
                if self.__peek(TokenType.DB):
                    self.__match(TokenType.DB)
                elif self.__peek(TokenType.DW):
                    self.__match(TokenType.DW)
                elif self.__peek(TokenType.DD):
                    self.__match(TokenType.DD)
                elif self.__peek(TokenType.DQ):
                    self.__match(TokenType.DQ)
                else:
                    exit(1)
                value.append(self.__lookahead.getLexeme())
                endPos = self.__lookahead.getEndPos()
                self.__match(TokenType.NUM)
                while self.__match(TokenType.COMMA):
                    self.__match(TokenType.COMMA)
                    value.append(self.__lookahead.getLexeme())
                    endPos = self.__lookahead.getEndPos()
                    self.__match(TokenType.NUM)

            return DataDeclaration(startPos, endPos, name, size, value, isConstant)

    # 4. <bss> := 'section' '.bss' <unit_decl>* ;
    def __bss(self):
        lst = list()
        self.__match(TokenType.SECTION)
        self.__match(TokenType.ID_BSS)
        while self.__peek(TokenType.ID):
            lst.append(self.__unitDecl())

        return lst

    # 5. <unit_decl> := <ID> (('resb' | 'resw' | 'resd' | 'resq') <number> ( ',' <number> )*)*;
    def __unitDecl(self):
        name = self.__lookahead.getLexeme()
        startPos = self.__lookahead.getStartPos()
        values = list()
        size = None
        self.__match(TokenType.ID)
        endPos = None
        while (
            self.__peek(TokenType.RESB)
            or self.__peek(TokenType.RESW)
            or self.__peek(TokenType.RESD)
            or self.__peek(TokenType.RESQ)
        ):
            size = self.__lookahead.getLexeme()
            if self.__peek(TokenType.RESB):
                self.__match(TokenType.RESB)
            elif self.__peek(TokenType.RESW):
                self.__match(TokenType.RESW)
            elif self.__peek(TokenType.RESD):
                self.__match(TokenType.RESD)
            elif self.__peek(TokenType.RESQ):
                self.__match(TokenType.RESQ)
            else:
                exit(1)

            values.append(self.__lookahead.getLexeme())
            self.__match(TokenType.NUM)
            endPos = self.__lookahead.getEndPos()
            while self.__match(TokenType.COMMA):
                self.__match(TokenType.COMMA)
                values.append(self.__lookahead.getLexeme())
                endPos = self.__lookahead.getEndPos()
                self.__match(TokenType.NUM)

        return UninitDeclaration(startPos, endPos, name, size, values)

    # 6. <text> := 'section' '.text' 'global' '_start' <ID> <code> ;
    def __text(self):
        self.__match(TokenType.SECTION)
        self.__match(TokenType.ID_TEXT)
        self.__match(TokenType.GLOBAL)
        self.__match(TokenType.START)
        self.__match(TokenType.ID)
        return self.__code()

    # 7. <code> := ( <func_decl> (<label> | <instr>)* )* ;
    def __code(self):
        lst = list()
        if self.__peek(TokenType.GLOBAL):
            lst.append(self.__funcDecl())

        while not self.__peek(TokenType.EOF):
            if self.__peek(TokenType.ID):
                lst.append(self.__label())
            else:
                lst.append(self.__instr())

        return lst

    # 8. <func_decl> := 'global' <ID> <label> ;
    def __funcDecl(self):
        self.__match(TokenType.GLOBAL)
        self.__match(TokenType.ID)
        return self.__label()

    # 9. <label> := <ID> ;
    def __label(self):
        label = self.__lookahead
        self.__match(TokenType.ID)
        return Label(label)

    # 10. <instr> := <instr_keyword> (<instr_info>)? ;
    def __instr(self):
        startPos = self.__lookahead.getStartPos()
        endPos = self.__lookahead.getEndPos()
        instr = self.__lookahead.getLexeme()
        self.__consume()
        if (
            self.__isRegister()
            or self.__peek(TokenType.BYTE)
            or self.__peek(TokenType.WORD)
            or self.__peek(TokenType.DWORD)
            or self.__peek(TokenType.QWORD)
        ):
            info: Token = self.__instrInfo()
            if info[len(info) - 1] is DataSize:
                endPos = info[len(info) - 1].getEndPos()
            else:
                endPos = info[len(info) - 1].getEndPos()
            return Instruction(startPos, endPos, instr, info)

        return Instruction(startPos, endPos, instr, list())

    # 11. <instr_info> := <reg_keyword> | <data_size> ;
    def __instrInfo(self):
        lst = list()
        while (
            self.__isRegister()
            or self.__peek(TokenType.BYTE)
            or self.__peek(TokenType.WORD)
            or self.__peek(TokenType.DWORD)
            or self.__peek(TokenType.QWORD)
        ):

            if (
                self.__peek(TokenType.BYTE)
                or self.__peek(TokenType.WORD)
                or self.__peek(TokenType.DWORD)
                or self.__peek(TokenType.QWORD)
            ):
                lst.append(self.__dataSize())
            else:
                lst.append(self.__lookahead)
                self.__consume()

            if self.__peek(TokenType.COMMA):
                self.__consume()
                if self.__peek(TokenType.NUM):
                    lst.append(self.__lookahead)
                    self.__consume()
            else:
                return lst
        return lst

    # 12. <data_size> := <size_keyword> '[' ( '(' | ')' | '+' | '-' | '*' | '%' | <reg_keyword> | <number> | <ID> )+ ']' ;
    def __dataSize(self):
        startPos = self.__lookahead.getStartPos()
        size = self.__lookahead.getLexeme()
        if self.__peek(TokenType.BYTE):
            self.__match(TokenType.BYTE)
        elif self.__peek(TokenType.WORD):
            self.__match(TokenType.WORD)
        elif self.__peek(TokenType.DWORD):
            self.__match(TokenType.DWORD)
        elif self.__peek(TokenType.QWORD):
            self.__match(TokenType.QWORD)
        else:
            exit(1)

        lst = list()
        self.__match(TokenType.LBRACK)
        while not self.__peek(TokenType.RBRACK):
            lst.append(self.__lookahead)
            if self.__peek(TokenType.LPAREN):
                self.__match(TokenType.LPAREN)
            elif self.__peek(TokenType.RPAREN):
                self.__match(TokenType.RPAREN)
            elif self.__peek(TokenType.PLUS):
                self.__match(TokenType.PLUS)
            elif self.__peek(TokenType.MINUS):
                self.__match(TokenType.MINUS)
            elif self.__peek(TokenType.MULTIPLY):
                self.__match(TokenType.MULTIPLY)
            elif self.__peek(TokenType.PERCENT):
                self.__match(TokenType.PERCENT)
            elif self.__peek(TokenType.NUM):
                self.__match(TokenType.NUM)
            elif self.__peek(TokenType.ID):
                self.__match(TokenType.ID)
            elif self.__isRegister():
                self.__consume()
            else:
                exit(1)
        endPos = self.__lookahead.getEndPos()
        self.__match(TokenType.RBRACK)

        return DataSize(startPos, endPos, size, lst)
