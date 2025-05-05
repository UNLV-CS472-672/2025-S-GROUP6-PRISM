"""
Created by Daniel Levy, 4/21/2025.

This is a Lexer class responsible for the tokenization of both
x86 and MIPS assembly files based on how assembly is taught to
CS218 students.
"""

from code_analysis.asm.asm_token.Position import Position
from code_analysis.asm.asm_token.Token import Token
from code_analysis.asm.asm_token.TokenType import TokenType


class Lexer:
    """Create object to perform tokenization."""

    __EOF: str = "\0"
    __program: str = None
    __currPos: int = None
    __lookahead: str = None
    __currLine: int = None
    __currCol: int = None

    def __init__(self, program):
        """Construct Lexer object."""
        self.__program = program
        self.__currPos = 0
        self.__lookahead = program[0]
        self.__currLine = 1
        self.__currCol = 1

    """
        This is a helper method to consume the current
        lookahead character and move on to the next character.
    """

    def __consume(self):
        self.__currPos += 1
        self.__currCol += 1

        if self.__currPos < len(self.__program):
            self.__lookahead = self.__program[self.__currPos]
        else:
            self.__lookahead = self.__EOF

    """
        This is a helper method responsible for checking if
        the current lookahead character matches a character
        we expect to see next.
    """

    def __match(self, currChar):
        if self.__lookahead == currChar:
            self.__consume()
            return True
        return False

    """
        This is a helper method responsible for consuming a
        comment, so it can be ignored and not added to the
        token array.
    """

    def __consumeComment(self):
        self.__consume()
        while self.__lookahead != "\n":
            self.__consume()
        self.__consumeNewLine()

    """
        This is a helper method that will consume a new line
        character for us and increment the current file position.
    """

    def __consumeNewLine(self):
        self.__consume()
        self.__currLine += 1
        self.__currCol = 1

    """
        This is a helper method that will ignore all whitespace.
    """

    def __consumeWhiteSpace(self):
        while (
            self.__lookahead == " "
            or self.__lookahead == "\t"
            or self.__lookahead == "\r"
        ):
            self.__consume()

    """
        This creates a new Position to be saved into a token.
    """

    def __createPosition(self):
        return Position(self.__currLine, self.__currCol)

    """
        This method is responisble for tokenizing a String literal.
    """

    def __tokenizeString(self):
        startPos = self.__createPosition()
        string = '"'
        self.__match('"')

        while not self.__match('"') and not self.__match(self.__EOF):
            string += self.__lookahead
            self.__consume()

        string += '"'
        return Token(TokenType.STR, string, startPos, self.__createPosition())

    """
        This method is responsible for tokenizing an Integer/Float literal.
    """

    def __tokenizeNumber(self):
        startPos = self.__createPosition()
        number = ""
        while self.__lookahead.isdigit():
            number += self.__lookahead
            self.__consume()
            if self.__lookahead == ".":
                number += self.__lookahead
                self.__consume()

        return Token(TokenType.NUM, number, startPos, self.__createPosition())

    """
        This is a method responsible for tokenizing all IDs and
        assembly keywords.
    """

    def __tokenizeName(self):
        startPos = self.__createPosition()
        id = ""
        while (
            self.__lookahead.isalpha()
            or self.__lookahead.isdigit()
            or self.__lookahead in ["%", "_", ".", "$"]
        ):
            id += self.__lookahead
            self.__consume()

        match id:
            # x86 Keywords
            case "rax":
                return Token(TokenType.RAX, id, startPos, self.__createPosition())
            case "eax":
                return Token(TokenType.EAX, id, startPos, self.__createPosition())
            case "ax":
                return Token(TokenType.AX, id, startPos, self.__createPosition())
            case "al":
                return Token(TokenType.AL, id, startPos, self.__createPosition())
            case "rbx":
                return Token(TokenType.RBX, id, startPos, self.__createPosition())
            case "ebx":
                return Token(TokenType.EBX, id, startPos, self.__createPosition())
            case "bx":
                return Token(TokenType.BX, id, startPos, self.__createPosition())
            case "bl":
                return Token(TokenType.BL, id, startPos, self.__createPosition())
            case "rcx":
                return Token(TokenType.RCX, id, startPos, self.__createPosition())
            case "ecx":
                return Token(TokenType.ECX, id, startPos, self.__createPosition())
            case "cx":
                return Token(TokenType.CX, id, startPos, self.__createPosition())
            case "cl":
                return Token(TokenType.CL, id, startPos, self.__createPosition())
            case "rdx":
                return Token(TokenType.RDX, id, startPos, self.__createPosition())
            case "edx":
                return Token(TokenType.EDX, id, startPos, self.__createPosition())
            case "dx":
                return Token(TokenType.DX, id, startPos, self.__createPosition())
            case "dl":
                return Token(TokenType.DL, id, startPos, self.__createPosition())
            case "rsi":
                return Token(TokenType.RSI, id, startPos, self.__createPosition())
            case "esi":
                return Token(TokenType.ESI, id, startPos, self.__createPosition())
            case "si":
                return Token(TokenType.SI, id, startPos, self.__createPosition())
            case "sil":
                return Token(TokenType.SIL, id, startPos, self.__createPosition())
            case "rdi":
                return Token(TokenType.RDI, id, startPos, self.__createPosition())
            case "edi":
                return Token(TokenType.EDI, id, startPos, self.__createPosition())
            case "di":
                return Token(TokenType.DI, id, startPos, self.__createPosition())
            case "dil":
                return Token(TokenType.DIL, id, startPos, self.__createPosition())
            case "rbp":
                return Token(TokenType.RBP, id, startPos, self.__createPosition())
            case "ebp":
                return Token(TokenType.EBP, id, startPos, self.__createPosition())
            case "bp":
                return Token(TokenType.BP, id, startPos, self.__createPosition())
            case "bpl":
                return Token(TokenType.BPL, id, startPos, self.__createPosition())
            case "rsp":
                return Token(TokenType.RSP, id, startPos, self.__createPosition())
            case "esp":
                return Token(TokenType.ESP, id, startPos, self.__createPosition())
            case "sp":
                return Token(TokenType.SP, id, startPos, self.__createPosition())
            case "spl":
                return Token(TokenType.SPL, id, startPos, self.__createPosition())
            case "r8":
                return Token(TokenType.R8, id, startPos, self.__createPosition())
            case "r8d":
                return Token(TokenType.R8D, id, startPos, self.__createPosition())
            case "r8w":
                return Token(TokenType.R8W, id, startPos, self.__createPosition())
            case "r8b":
                return Token(TokenType.R8B, id, startPos, self.__createPosition())
            case "r9":
                return Token(TokenType.R9, id, startPos, self.__createPosition())
            case "r9d":
                return Token(TokenType.R9D, id, startPos, self.__createPosition())
            case "r9w":
                return Token(TokenType.R9W, id, startPos, self.__createPosition())
            case "r9b":
                return Token(TokenType.R9B, id, startPos, self.__createPosition())
            case "r10":
                return Token(TokenType.R10, id, startPos, self.__createPosition())
            case "r10d":
                return Token(TokenType.R10D, id, startPos, self.__createPosition())
            case "r10w":
                return Token(TokenType.R10W, id, startPos, self.__createPosition())
            case "r10b":
                return Token(TokenType.R10B, id, startPos, self.__createPosition())
            case "r11":
                return Token(TokenType.R11, id, startPos, self.__createPosition())
            case "r11d":
                return Token(TokenType.R11D, id, startPos, self.__createPosition())
            case "r11w":
                return Token(TokenType.R11W, id, startPos, self.__createPosition())
            case "r11b":
                return Token(TokenType.R11B, id, startPos, self.__createPosition())
            case "r12":
                return Token(TokenType.R12, id, startPos, self.__createPosition())
            case "r12d":
                return Token(TokenType.R12D, id, startPos, self.__createPosition())
            case "r12w":
                return Token(TokenType.R12W, id, startPos, self.__createPosition())
            case "r12b":
                return Token(TokenType.R12B, id, startPos, self.__createPosition())
            case "r13":
                return Token(TokenType.R13, id, startPos, self.__createPosition())
            case "r13d":
                return Token(TokenType.R13D, id, startPos, self.__createPosition())
            case "r13w":
                return Token(TokenType.R13W, id, startPos, self.__createPosition())
            case "r13b":
                return Token(TokenType.R13B, id, startPos, self.__createPosition())
            case "r14":
                return Token(TokenType.R14, id, startPos, self.__createPosition())
            case "r14d":
                return Token(TokenType.R14D, id, startPos, self.__createPosition())
            case "r14w":
                return Token(TokenType.R14W, id, startPos, self.__createPosition())
            case "r14b":
                return Token(TokenType.R14B, id, startPos, self.__createPosition())
            case "r15":
                return Token(TokenType.R15, id, startPos, self.__createPosition())
            case "r15d":
                return Token(TokenType.R15D, id, startPos, self.__createPosition())
            case "r15w":
                return Token(TokenType.R15W, id, startPos, self.__createPosition())
            case "r15b":
                return Token(TokenType.R15B, id, startPos, self.__createPosition())
            case "xmm0":
                return Token(TokenType.XMM0, id, startPos, self.__createPosition())
            case "xmm1":
                return Token(TokenType.XMM1, id, startPos, self.__createPosition())
            case "xmm2":
                return Token(TokenType.XMM2, id, startPos, self.__createPosition())
            case "xmm3":
                return Token(TokenType.XMM3, id, startPos, self.__createPosition())
            case "xmm4":
                return Token(TokenType.XMM4, id, startPos, self.__createPosition())
            case "xmm5":
                return Token(TokenType.XMM5, id, startPos, self.__createPosition())
            case "xmm6":
                return Token(TokenType.XMM6, id, startPos, self.__createPosition())
            case "xmm7":
                return Token(TokenType.XMM7, id, startPos, self.__createPosition())
            case "xmm8":
                return Token(TokenType.XMM8, id, startPos, self.__createPosition())
            case "xmm9":
                return Token(TokenType.XMM9, id, startPos, self.__createPosition())
            case "xmm10":
                return Token(TokenType.XMM10, id, startPos, self.__createPosition())
            case "xmm11":
                return Token(TokenType.XMM11, id, startPos, self.__createPosition())
            case "xmm12":
                return Token(TokenType.XMM12, id, startPos, self.__createPosition())
            case "xmm13":
                return Token(TokenType.XMM13, id, startPos, self.__createPosition())
            case "xmm14":
                return Token(TokenType.XMM14, id, startPos, self.__createPosition())
            case "xmm15":
                return Token(TokenType.XMM15, id, startPos, self.__createPosition())

            case "equ":
                return Token(TokenType.EQU, id, startPos, self.__createPosition())
            case "db":
                return Token(TokenType.DB, id, startPos, self.__createPosition())
            case "dw":
                return Token(TokenType.DW, id, startPos, self.__createPosition())
            case "dd":
                return Token(TokenType.DD, id, startPos, self.__createPosition())
            case "dq":
                return Token(TokenType.DQ, id, startPos, self.__createPosition())
            case "ddq":
                return Token(TokenType.DDQ, id, startPos, self.__createPosition())
            case "dt":
                return Token(TokenType.DT, id, startPos, self.__createPosition())
            case "resb":
                return Token(TokenType.RESB, id, startPos, self.__createPosition())
            case "resw":
                return Token(TokenType.RESW, id, startPos, self.__createPosition())
            case "resd":
                return Token(TokenType.RESD, id, startPos, self.__createPosition())
            case "resq":
                return Token(TokenType.RESQ, id, startPos, self.__createPosition())
            case "resdq":
                return Token(TokenType.RESDQ, id, startPos, self.__createPosition())
            case "byte":
                return Token(TokenType.BYTE, id, startPos, self.__createPosition())
            case "word":
                return Token(TokenType.WORD, id, startPos, self.__createPosition())
            case "dword":
                return Token(TokenType.DWORD, id, startPos, self.__createPosition())
            case "qword":
                return Token(TokenType.QWORD, id, startPos, self.__createPosition())
            case "section":
                return Token(TokenType.SECTION, id, startPos, self.__createPosition())
            case "global":
                return Token(TokenType.GLOBAL, id, startPos, self.__createPosition())
            case ".data":
                return Token(TokenType.ID_DATA, id, startPos, self.__createPosition())
            case ".bss":
                return Token(TokenType.ID_BSS, id, startPos, self.__createPosition())
            case ".text":
                return Token(TokenType.ID_TEXT, id, startPos, self.__createPosition())
            case "_start":
                return Token(TokenType.START, id, startPos, self.__createPosition())
            case "mov":
                return Token(TokenType.MOV, id, startPos, self.__createPosition())
            case "lea":
                return Token(TokenType.LEA, id, startPos, self.__createPosition())
            case "movzx":
                return Token(TokenType.MOVZX, id, startPos, self.__createPosition())
            case "cbw":
                return Token(TokenType.CBW, id, startPos, self.__createPosition())
            case "cwd":
                return Token(TokenType.CWD, id, startPos, self.__createPosition())
            case "cwde":
                return Token(TokenType.CWDE, id, startPos, self.__createPosition())
            case "cdq":
                return Token(TokenType.CDQ, id, startPos, self.__createPosition())
            case "cdqe":
                return Token(TokenType.CDQE, id, startPos, self.__createPosition())
            case "cqo":
                return Token(TokenType.CQO, id, startPos, self.__createPosition())
            case "movsx":
                return Token(TokenType.MOVSX, id, startPos, self.__createPosition())
            case "movsxd":
                return Token(TokenType.MOVSXD, id, startPos, self.__createPosition())
            case "add":
                return Token(TokenType.ADD, id, startPos, self.__createPosition())
            case "inc":
                return Token(TokenType.INC, id, startPos, self.__createPosition())
            case "adc":
                return Token(TokenType.ADC, id, startPos, self.__createPosition())
            case "sub":
                return Token(TokenType.SUB, id, startPos, self.__createPosition())
            case "dec":
                return Token(TokenType.DEC, id, startPos, self.__createPosition())
            case "mul":
                return Token(TokenType.MUL, id, startPos, self.__createPosition())
            case "imul":
                return Token(TokenType.IMUL, id, startPos, self.__createPosition())
            case "div":
                return Token(TokenType.DIV, id, startPos, self.__createPosition())
            case "idiv":
                return Token(TokenType.IDIV, id, startPos, self.__createPosition())
            case "and":
                return Token(TokenType.AND, id, startPos, self.__createPosition())
            case "or":
                return Token(TokenType.OR, id, startPos, self.__createPosition())
            case "xor":
                return Token(TokenType.XOR, id, startPos, self.__createPosition())
            case "not":
                return Token(TokenType.NOT, id, startPos, self.__createPosition())
            case "shl":
                return Token(TokenType.SHL, id, startPos, self.__createPosition())
            case "shr":
                return Token(TokenType.SHR, id, startPos, self.__createPosition())
            case "sal":
                return Token(TokenType.SAL, id, startPos, self.__createPosition())
            case "sar":
                return Token(TokenType.SAR, id, startPos, self.__createPosition())
            case "rol":
                return Token(TokenType.ROL, id, startPos, self.__createPosition())
            case "ror":
                return Token(TokenType.ROR, id, startPos, self.__createPosition())
            case "jmp":
                return Token(TokenType.JMP, id, startPos, self.__createPosition())
            case "cmp":
                return Token(TokenType.CMP, id, startPos, self.__createPosition())
            case "je":
                return Token(TokenType.JE, id, startPos, self.__createPosition())
            case "jne":
                return Token(TokenType.JNE, id, startPos, self.__createPosition())
            case "jl":
                return Token(TokenType.JL, id, startPos, self.__createPosition())
            case "jle":
                return Token(TokenType.JLE, id, startPos, self.__createPosition())
            case "jg":
                return Token(TokenType.JG, id, startPos, self.__createPosition())
            case "jge":
                return Token(TokenType.JGE, id, startPos, self.__createPosition())
            case "jb":
                return Token(TokenType.JB, id, startPos, self.__createPosition())
            case "jbe":
                return Token(TokenType.JBE, id, startPos, self.__createPosition())
            case "ja":
                return Token(TokenType.JA, id, startPos, self.__createPosition())
            case "jae":
                return Token(TokenType.JAE, id, startPos, self.__createPosition())
            case "loop":
                return Token(TokenType.LOOP, id, startPos, self.__createPosition())
            case "push":
                return Token(TokenType.PUSH, id, startPos, self.__createPosition())
            case "pop":
                return Token(TokenType.POP, id, startPos, self.__createPosition())
            case "call":
                return Token(TokenType.CALL, id, startPos, self.__createPosition())
            case "ret":
                return Token(TokenType.RET, id, startPos, self.__createPosition())
            case "movss":
                return Token(TokenType.MOVSS, id, startPos, self.__createPosition())
            case "movsd":
                return Token(TokenType.MOVSD, id, startPos, self.__createPosition())
            case "cvtss2sd":
                return Token(TokenType.CVTSS2SD, id, startPos, self.__createPosition())
            case "cvtsd2ss":
                return Token(TokenType.CVTSD2SS, id, startPos, self.__createPosition())
            case "cvtss2si":
                return Token(TokenType.CVTSS2SI, id, startPos, self.__createPosition())
            case "cvtsd2si":
                return Token(TokenType.CVTSD2SI, id, startPos, self.__createPosition())
            case "cvtsi2ss":
                return Token(TokenType.CVTSI2SS, id, startPos, self.__createPosition())
            case "cvtsi2sd":
                return Token(TokenType.CVTSI2SD, id, startPos, self.__createPosition())
            case "addss":
                return Token(TokenType.ADDSS, id, startPos, self.__createPosition())
            case "addsd":
                return Token(TokenType.ADDSD, id, startPos, self.__createPosition())
            case "subss":
                return Token(TokenType.SUBSS, id, startPos, self.__createPosition())
            case "subsd":
                return Token(TokenType.SUBSD, id, startPos, self.__createPosition())
            case "mulss":
                return Token(TokenType.MULSS, id, startPos, self.__createPosition())
            case "mulsd":
                return Token(TokenType.MULSD, id, startPos, self.__createPosition())
            case "divss":
                return Token(TokenType.DIVSS, id, startPos, self.__createPosition())
            case "divsd":
                return Token(TokenType.DIVSD, id, startPos, self.__createPosition())
            case "sqrtss":
                return Token(TokenType.SQRTSS, id, startPos, self.__createPosition())
            case "sqrtsd":
                return Token(TokenType.SQRTSD, id, startPos, self.__createPosition())
            case "ucomiss":
                return Token(TokenType.UCOMIS, id, startPos, self.__createPosition())
            case "ucomisd":
                return Token(TokenType.UCOMISD, id, startPos, self.__createPosition())
            case "syscall":
                return Token(TokenType.SYSCALL, id, startPos, self.__createPosition())
            case "cmovl":
                return Token(TokenType.CMOVL, id, startPos, self.__createPosition())
            case "cmovg":
                return Token(TokenType.CMOVG, id, startPos, self.__createPosition())
            case "asr":
                return Token(TokenType.ASR, id, startPos, self.__createPosition())
            case "jnz":
                return Token(TokenType.JNZ, id, startPos, self.__createPosition())
            case "rel":
                return Token(TokenType.REL, id, startPos, self.__createPosition())
            case "%define":
                return Token(TokenType.DEFINE, id, startPos, self.__createPosition())

            # MIPS Keywords
            case "$zero":
                return Token(TokenType.ZERO, id, startPos, self.__createPosition())
            case "$at":
                return Token(TokenType.AT, id, startPos, self.__createPosition())
            case "$v0":
                return Token(TokenType.V0, id, startPos, self.__createPosition())
            case "$v1":
                return Token(TokenType.V1, id, startPos, self.__createPosition())
            case "$a0":
                return Token(TokenType.A0, id, startPos, self.__createPosition())
            case "$a1":
                return Token(TokenType.A1, id, startPos, self.__createPosition())
            case "$a2":
                return Token(TokenType.A2, id, startPos, self.__createPosition())
            case "$a3":
                return Token(TokenType.A3, id, startPos, self.__createPosition())
            case "$t0":
                return Token(TokenType.T0, id, startPos, self.__createPosition())
            case "$t1":
                return Token(TokenType.T1, id, startPos, self.__createPosition())
            case "$t2":
                return Token(TokenType.T2, id, startPos, self.__createPosition())
            case "$t3":
                return Token(TokenType.T3, id, startPos, self.__createPosition())
            case "$t4":
                return Token(TokenType.T4, id, startPos, self.__createPosition())
            case "$t5":
                return Token(TokenType.T5, id, startPos, self.__createPosition())
            case "$t6":
                return Token(TokenType.T6, id, startPos, self.__createPosition())
            case "$t7":
                return Token(TokenType.T7, id, startPos, self.__createPosition())
            case "$s0":
                return Token(TokenType.S0, id, startPos, self.__createPosition())
            case "$s1":
                return Token(TokenType.S1, id, startPos, self.__createPosition())
            case "$s2":
                return Token(TokenType.S2, id, startPos, self.__createPosition())
            case "$s3":
                return Token(TokenType.S3, id, startPos, self.__createPosition())
            case "$s4":
                return Token(TokenType.S4, id, startPos, self.__createPosition())
            case "$s5":
                return Token(TokenType.S5, id, startPos, self.__createPosition())
            case "$s6":
                return Token(TokenType.S6, id, startPos, self.__createPosition())
            case "$s7":
                return Token(TokenType.S7, id, startPos, self.__createPosition())
            case "$t8":
                return Token(TokenType.T8, id, startPos, self.__createPosition())
            case "$t9":
                return Token(TokenType.T9, id, startPos, self.__createPosition())
            case "$k0":
                return Token(TokenType.K0, id, startPos, self.__createPosition())
            case "$k1":
                return Token(TokenType.K1, id, startPos, self.__createPosition())
            case "$gp":
                return Token(TokenType.GP, id, startPos, self.__createPosition())
            case "$sp":
                return Token(TokenType.MIPS_SP, id, startPos, self.__createPosition())
            case "$fp":
                return Token(TokenType.FP, id, startPos, self.__createPosition())
            case "$ra":
                return Token(TokenType.RA, id, startPos, self.__createPosition())
            case "$hi":
                return Token(TokenType.HI, id, startPos, self.__createPosition())
            case "$lo":
                return Token(TokenType.LO, id, startPos, self.__createPosition())
            case "$f0":
                return Token(TokenType.F0, id, startPos, self.__createPosition())
            case "$f1":
                return Token(TokenType.F1, id, startPos, self.__createPosition())
            case "$f2":
                return Token(TokenType.F2, id, startPos, self.__createPosition())
            case "$f3":
                return Token(TokenType.F3, id, startPos, self.__createPosition())
            case "$f4":
                return Token(TokenType.F4, id, startPos, self.__createPosition())
            case "$f5":
                return Token(TokenType.F5, id, startPos, self.__createPosition())
            case "$f6":
                return Token(TokenType.F6, id, startPos, self.__createPosition())
            case "$f7":
                return Token(TokenType.F7, id, startPos, self.__createPosition())
            case "$f8":
                return Token(TokenType.F8, id, startPos, self.__createPosition())
            case "$f9":
                return Token(TokenType.F9, id, startPos, self.__createPosition())
            case "$f10":
                return Token(TokenType.F10, id, startPos, self.__createPosition())
            case "$f11":
                return Token(TokenType.F11, id, startPos, self.__createPosition())
            case "$f12":
                return Token(TokenType.F12, id, startPos, self.__createPosition())
            case "$f13":
                return Token(TokenType.F13, id, startPos, self.__createPosition())
            case "$f14":
                return Token(TokenType.F14, id, startPos, self.__createPosition())
            case "$f15":
                return Token(TokenType.F15, id, startPos, self.__createPosition())
            case "$f16":
                return Token(TokenType.F16, id, startPos, self.__createPosition())
            case "$f17":
                return Token(TokenType.F17, id, startPos, self.__createPosition())
            case "$f18":
                return Token(TokenType.F18, id, startPos, self.__createPosition())
            case "$f19":
                return Token(TokenType.F19, id, startPos, self.__createPosition())
            case "$f20":
                return Token(TokenType.F20, id, startPos, self.__createPosition())
            case "$f21":
                return Token(TokenType.F21, id, startPos, self.__createPosition())
            case "$f22":
                return Token(TokenType.F22, id, startPos, self.__createPosition())
            case "$f23":
                return Token(TokenType.F23, id, startPos, self.__createPosition())
            case "$f24":
                return Token(TokenType.F24, id, startPos, self.__createPosition())
            case "$f25":
                return Token(TokenType.F25, id, startPos, self.__createPosition())
            case "$f26":
                return Token(TokenType.F26, id, startPos, self.__createPosition())
            case "$f27":
                return Token(TokenType.F27, id, startPos, self.__createPosition())
            case "$f28":
                return Token(TokenType.F28, id, startPos, self.__createPosition())
            case "$f29":
                return Token(TokenType.F29, id, startPos, self.__createPosition())
            case "$f30":
                return Token(TokenType.F30, id, startPos, self.__createPosition())
            case "$f31":
                return Token(TokenType.F31, id, startPos, self.__createPosition())

            case "half":
                return Token(TokenType.HALF, id, startPos, self.__createPosition())
            case ".ascii":
                return Token(TokenType.ASCII, id, startPos, self.__createPosition())
            case ".asciiz":
                return Token(TokenType.ASCIIZ, id, startPos, self.__createPosition())
            case "float":
                return Token(TokenType.FLOAT, id, startPos, self.__createPosition())
            case "double":
                return Token(TokenType.DOUBLE, id, startPos, self.__createPosition())
            case "space":
                return Token(TokenType.SPACE, id, startPos, self.__createPosition())
            case "abs":
                return Token(TokenType.ABS, id, startPos, self.__createPosition())
            case "addu":
                return Token(TokenType.ADDU, id, startPos, self.__createPosition())
            case "divu":
                return Token(TokenType.DIVU, id, startPos, self.__createPosition())
            case "mulo":
                return Token(TokenType.MULO, id, startPos, self.__createPosition())
            case "mulou":
                return Token(TokenType.MULOU, id, startPos, self.__createPosition())
            case "mult":
                return Token(TokenType.MULT, id, startPos, self.__createPosition())
            case "multu":
                return Token(TokenType.MULTU, id, startPos, self.__createPosition())
            case "neg":
                return Token(TokenType.NEG, id, startPos, self.__createPosition())
            case "rem":
                return Token(TokenType.REM, id, startPos, self.__createPosition())
            case "remu":
                return Token(TokenType.REMU, id, startPos, self.__createPosition())
            case "subu":
                return Token(TokenType.SUBU, id, startPos, self.__createPosition())
            case "seq":
                return Token(TokenType.SEQ, id, startPos, self.__createPosition())
            case "sge":
                return Token(TokenType.SGE, id, startPos, self.__createPosition())
            case "sgeu":
                return Token(TokenType.SGEU, id, startPos, self.__createPosition())
            case "sgt":
                return Token(TokenType.SGT, id, startPos, self.__createPosition())
            case "sgtu":
                return Token(TokenType.SGTU, id, startPos, self.__createPosition())
            case "sle":
                return Token(TokenType.SLE, id, startPos, self.__createPosition())
            case "sleu":
                return Token(TokenType.SLEU, id, startPos, self.__createPosition())
            case "slt":
                return Token(TokenType.SLT, id, startPos, self.__createPosition())
            case "slti":
                return Token(TokenType.SLTI, id, startPos, self.__createPosition())
            case "sltu":
                return Token(TokenType.SLTU, id, startPos, self.__createPosition())
            case "sltiu":
                return Token(TokenType.SLTIU, id, startPos, self.__createPosition())
            case "sne":
                return Token(TokenType.SNE, id, startPos, self.__createPosition())
            case "b":
                return Token(TokenType.BRANCH, id, startPos, self.__createPosition())
            case "bczt":
                return Token(TokenType.BCZT, id, startPos, self.__createPosition())
            case "bczf":
                return Token(TokenType.BCZF, id, startPos, self.__createPosition())
            case "beq":
                return Token(TokenType.BEQ, id, startPos, self.__createPosition())
            case "beqz":
                return Token(TokenType.BEQZ, id, startPos, self.__createPosition())
            case "bge":
                return Token(TokenType.BGE, id, startPos, self.__createPosition())
            case "bgeu":
                return Token(TokenType.BGEU, id, startPos, self.__createPosition())
            case "bgez":
                return Token(TokenType.BGEZ, id, startPos, self.__createPosition())
            case "bgezal":
                return Token(TokenType.BGEZAL, id, startPos, self.__createPosition())
            case "bgt":
                return Token(TokenType.BGT, id, startPos, self.__createPosition())
            case "bgtu":
                return Token(TokenType.BGTU, id, startPos, self.__createPosition())
            case "bgtz":
                return Token(TokenType.BGTZ, id, startPos, self.__createPosition())
            case "ble":
                return Token(TokenType.BLE, id, startPos, self.__createPosition())
            case "bleu":
                return Token(TokenType.BLEU, id, startPos, self.__createPosition())
            case "blez":
                return Token(TokenType.BLEZ, id, startPos, self.__createPosition())
            case "blezal":
                return Token(TokenType.BLEZAL, id, startPos, self.__createPosition())
            case "bltzal":
                return Token(TokenType.BLTZAL, id, startPos, self.__createPosition())
            case "blt":
                return Token(TokenType.BLT, id, startPos, self.__createPosition())
            case "bltu":
                return Token(TokenType.BLTU, id, startPos, self.__createPosition())
            case "bltz":
                return Token(TokenType.BLTZ, id, startPos, self.__createPosition())
            case "bne":
                return Token(TokenType.BNE, id, startPos, self.__createPosition())
            case "bnez":
                return Token(TokenType.BNEZ, id, startPos, self.__createPosition())
            case "j":
                return Token(TokenType.JUMP, id, startPos, self.__createPosition())
            case "jal":
                return Token(TokenType.JAL, id, startPos, self.__createPosition())
            case "jalr":
                return Token(TokenType.JALR, id, startPos, self.__createPosition())
            case "jr":
                return Token(TokenType.JR, id, startPos, self.__createPosition())
            case "la":
                return Token(TokenType.LA, id, startPos, self.__createPosition())
            case "lb":
                return Token(TokenType.LB, id, startPos, self.__createPosition())
            case "lbu":
                return Token(TokenType.LBU, id, startPos, self.__createPosition())
            case "ld":
                return Token(TokenType.LD, id, startPos, self.__createPosition())
            case "lh":
                return Token(TokenType.LH, id, startPos, self.__createPosition())
            case "lhu":
                return Token(TokenType.LHU, id, startPos, self.__createPosition())
            case "lw":
                return Token(TokenType.LW, id, startPos, self.__createPosition())
            case "lwcz":
                return Token(TokenType.LWCZ, id, startPos, self.__createPosition())
            case "lwl":
                return Token(TokenType.LWL, id, startPos, self.__createPosition())
            case "lwr":
                return Token(TokenType.LWR, id, startPos, self.__createPosition())
            case "ulh":
                return Token(TokenType.ULH, id, startPos, self.__createPosition())
            case "ulhu":
                return Token(TokenType.ULHU, id, startPos, self.__createPosition())
            case "ulw":
                return Token(TokenType.ULW, id, startPos, self.__createPosition())
            case "li":
                return Token(TokenType.LI, id, startPos, self.__createPosition())
            case "lui":
                return Token(TokenType.LUI, id, startPos, self.__createPosition())
            case "andi":
                return Token(TokenType.ANDI, id, startPos, self.__createPosition())
            case "nor":
                return Token(TokenType.NOR, id, startPos, self.__createPosition())
            case "ori":
                return Token(TokenType.ORI, id, startPos, self.__createPosition())
            case "sll":
                return Token(TokenType.SLL, id, startPos, self.__createPosition())
            case "sra":
                return Token(TokenType.SRA, id, startPos, self.__createPosition())
            case "srl":
                return Token(TokenType.SRL, id, startPos, self.__createPosition())
            case "xori":
                return Token(TokenType.XORI, id, startPos, self.__createPosition())
            case "sb":
                return Token(TokenType.SB, id, startPos, self.__createPosition())
            case "sd":
                return Token(TokenType.SD, id, startPos, self.__createPosition())
            case "sh":
                return Token(TokenType.SH, id, startPos, self.__createPosition())
            case "sw":
                return Token(TokenType.SW, id, startPos, self.__createPosition())
            case "swcz":
                return Token(TokenType.SWCZ, id, startPos, self.__createPosition())
            case "swl":
                return Token(TokenType.SWL, id, startPos, self.__createPosition())
            case "swr":
                return Token(TokenType.SWR, id, startPos, self.__createPosition())
            case "ush":
                return Token(TokenType.USH, id, startPos, self.__createPosition())
            case "usw":
                return Token(TokenType.USW, id, startPos, self.__createPosition())
            case "move":
                return Token(TokenType.MOVE, id, startPos, self.__createPosition())
            case "mfhi":
                return Token(TokenType.MFHI, id, startPos, self.__createPosition())
            case "mflo":
                return Token(TokenType.MFLO, id, startPos, self.__createPosition())
            case "mthi":
                return Token(TokenType.MTHI, id, startPos, self.__createPosition())
            case "mtlo":
                return Token(TokenType.MTLO, id, startPos, self.__createPosition())
            case "mfc1":
                return Token(TokenType.MFC1, id, startPos, self.__createPosition())
            case "mfc1.d":
                return Token(TokenType.MFC1D, id, startPos, self.__createPosition())
            case "mtc1":
                return Token(TokenType.MTC1, id, startPos, self.__createPosition())
            case "mtc1.d":
                return Token(TokenType.MTC1D, id, startPos, self.__createPosition())
            case "abs.d":
                return Token(TokenType.ABSD, id, startPos, self.__createPosition())
            case "abs.s":
                return Token(TokenType.ABSS, id, startPos, self.__createPosition())
            case "add.d":
                return Token(TokenType.ADDD, id, startPos, self.__createPosition())
            case "add.s":
                return Token(TokenType.ADDS, id, startPos, self.__createPosition())
            case "c.eq.d":
                return Token(TokenType.CEQD, id, startPos, self.__createPosition())
            case "c.eq.s":
                return Token(TokenType.CEQS, id, startPos, self.__createPosition())
            case "c.le.d":
                return Token(TokenType.CLED, id, startPos, self.__createPosition())
            case "c.le.s":
                return Token(TokenType.CLES, id, startPos, self.__createPosition())
            case "c.lt.d":
                return Token(TokenType.CLTD, id, startPos, self.__createPosition())
            case "c.lt.s":
                return Token(TokenType.CLTS, id, startPos, self.__createPosition())
            case "cvt.d.s":
                return Token(TokenType.CVTDS, id, startPos, self.__createPosition())
            case "cvt.d.w":
                return Token(TokenType.CVTDW, id, startPos, self.__createPosition())
            case "cvt.s.d":
                return Token(TokenType.CVTSD, id, startPos, self.__createPosition())
            case "cvt.s.w":
                return Token(TokenType.CVTSW, id, startPos, self.__createPosition())
            case "cvt.w.d":
                return Token(TokenType.CVTWD, id, startPos, self.__createPosition())
            case "cvt.w.s":
                return Token(TokenType.CVTWS, id, startPos, self.__createPosition())
            case "div.d":
                return Token(TokenType.DIVD, id, startPos, self.__createPosition())
            case "div.s":
                return Token(TokenType.DIVS, id, startPos, self.__createPosition())
            case "l.d":
                return Token(TokenType.LDD, id, startPos, self.__createPosition())
            case "l.s":
                return Token(TokenType.LDS, id, startPos, self.__createPosition())
            case "mov.d":
                return Token(TokenType.MOVD, id, startPos, self.__createPosition())
            case "mov.s":
                return Token(TokenType.MOVS, id, startPos, self.__createPosition())
            case "mul.d":
                return Token(TokenType.MULD, id, startPos, self.__createPosition())
            case "mul.s":
                return Token(TokenType.MULS, id, startPos, self.__createPosition())
            case "neg.d":
                return Token(TokenType.NEGD, id, startPos, self.__createPosition())
            case "neg.s":
                return Token(TokenType.NEGS, id, startPos, self.__createPosition())
            case "s.d":
                return Token(TokenType.SDD, id, startPos, self.__createPosition())
            case "s.s":
                return Token(TokenType.SDS, id, startPos, self.__createPosition())
            case "sub.d":
                return Token(TokenType.SUBD, id, startPos, self.__createPosition())
            case "sub.s":
                return Token(TokenType.SUBS, id, startPos, self.__createPosition())
            case "rfe":
                return Token(TokenType.RFE, id, startPos, self.__createPosition())
            case "break":
                return Token(TokenType.BREAK, id, startPos, self.__createPosition())
            case "nop":
                return Token(TokenType.NOP, id, startPos, self.__createPosition())
            case _:
                return Token(TokenType.ID, id, startPos, self.__createPosition())

    """
        nextToken is the main method that a user will interface with this class, and
        it is responsible for generating the next token from the input file. If there
        is an error, then an error token will be generated instead.
    """

    def nextToken(self):
        """Generate next token from student submission."""
        while self.__lookahead != self.__EOF:
            startPos = self.__createPosition()
            match self.__lookahead:
                case ";" | "#":
                    self.__consumeComment()
                case "\n":
                    self.__consumeNewLine()
                case " " | "\t" | "\r":
                    self.__consumeWhiteSpace()
                case "=":
                    self.__match("=")
                    return Token(TokenType.EQ, "=", startPos, self.__createPosition())
                case "+":
                    self.__match("+")
                    return Token(TokenType.PLUS, "+", startPos, self.__createPosition())
                case "-":
                    self.__match("-")
                    return Token(
                        TokenType.MINUS, "-", startPos, self.__createPosition()
                    )
                case "*":
                    self.__match("*")
                    return Token(
                        TokenType.MULTIPLY, "*", startPos, self.__createPosition()
                    )
                case "/":
                    self.__match("/")
                    return Token(
                        TokenType.DIVIDE, "/", startPos, self.__createPosition()
                    )
                case "%":
                    self.__match("%")
                    return Token(TokenType.MOD, "%", startPos, self.__createPosition())
                case "|":
                    self.__match("|")
                    return Token(TokenType.BOR, "|", startPos, self.__createPosition())
                case "&":
                    self.__match("&")
                    return Token(TokenType.BAND, "&", startPos, self.__createPosition())
                case ".":
                    self.__match(".")
                    if self.__lookahead.isalpha():
                        self.__currPos -= 1
                        self.__lookahead = "."
                        return self.__tokenizeName()
                    return Token(
                        TokenType.PERIOD, ".", startPos, self.__createPosition()
                    )
                case ",":
                    self.__match(",")
                    return Token(
                        TokenType.COMMA, ",", startPos, self.__createPosition()
                    )
                case "(":
                    self.__match("(")
                    return Token(
                        TokenType.LPAREN, "(", startPos, self.__createPosition()
                    )
                case ")":
                    self.__match(")")
                    return Token(
                        TokenType.RPAREN, ")", startPos, self.__createPosition()
                    )
                case "[":
                    self.__match("[")
                    return Token(
                        TokenType.LBRACK, "[", startPos, self.__createPosition()
                    )
                case "]":
                    self.__match("]")
                    return Token(
                        TokenType.RBRACK, "]", startPos, self.__createPosition()
                    )
                case ":":
                    self.__match(":")
                    return Token(
                        TokenType.COLON, ":", startPos, self.__createPosition()
                    )
                case "'":
                    self.__match("'")
                    return Token(
                        TokenType.SQUOTE, "'", startPos, self.__createPosition()
                    )
                case '"':
                    return self.__tokenizeString()
                case _:
                    if self.__lookahead.isdigit():
                        return self.__tokenizeNumber()
                    elif (
                        self.__lookahead.isalpha()
                        or self.__lookahead == "%"
                        or self.__lookahead == "_"
                        or self.__lookahead == "$"
                    ):
                        return self.__tokenizeName()
                    else:
                        return Token(
                            TokenType.ERROR, "ERROR!", startPos, self.__createPosition()
                        )

        return Token(
            TokenType.EOF, "$", self.__createPosition(), self.__createPosition()
        )
