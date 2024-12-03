# code sequence features
NUMERIC_CONSTANT = 0 # acho que não tem nenhum deste tipo
TRANSFER_INSTRUCTION = 1
CALL_INSTRUCTION = 2
ARITHMETIC_INSTRUCTION = 3
COMPARE_INSTRUCTION = 4
MOV_INSTRUCTION = 5
TERMINATION_INSTRUCTION = 6
DATA_DECLARATION_INSTRUCTION = 7

# memory management functions
ALLOCATION_FUNCTIONS = 0
DEALLOCATION_FUNCTIONS = 1
POINTER_ASSIGNMENT = 2
MEMORY_ADDRESS_OF = 3
CONVERT_UNSAFE = 4
STRING_UNSAFE = 5
SCANF_UNSAFE = 6
OTHER_UNSAFE = 7

statement_type_map = {
    "<operator>.addition": ARITHMETIC_INSTRUCTION,
    #"<operator>.addressOf": TRANSFER_INSTRUCTION,
    "<operator>.and": ARITHMETIC_INSTRUCTION,
    "<operator>.arithmeticShiftRight": ARITHMETIC_INSTRUCTION,
    "<operator>.assignment": MOV_INSTRUCTION,
    "<operator>.assignmentDivision": ARITHMETIC_INSTRUCTION,
    "<operator>.assignmentMinus": ARITHMETIC_INSTRUCTION,
    "<operator>.assignmentMultiplication": ARITHMETIC_INSTRUCTION,
    "<operator>.assignmentPlus": ARITHMETIC_INSTRUCTION,
    "<operator>.cast": TRANSFER_INSTRUCTION,
    "<operator>.conditional": COMPARE_INSTRUCTION,
    "<operator>.delete": MOV_INSTRUCTION,
    "<operator>.division": ARITHMETIC_INSTRUCTION,
    "<operator>.equals": COMPARE_INSTRUCTION,
    "<operator>.fieldAccess": MOV_INSTRUCTION,
    "<operator>.greaterEqualsThan": COMPARE_INSTRUCTION,
    "<operator>.greaterThan": COMPARE_INSTRUCTION,
    "<operator>.indirectFieldAccess": MOV_INSTRUCTION,
    "<operator>.indirectIndexAccess": MOV_INSTRUCTION,
    "<operator>.lessEqualsThan": COMPARE_INSTRUCTION,
    "<operator>.lessThan": COMPARE_INSTRUCTION,
    "<operator>.logicalAnd": COMPARE_INSTRUCTION,
    "<operator>.logicalNot": COMPARE_INSTRUCTION,
    "<operator>.logicalOr": COMPARE_INSTRUCTION,
    "<operator>.minus": ARITHMETIC_INSTRUCTION,
    "<operator>.modulo": ARITHMETIC_INSTRUCTION,
    "<operator>.multiplication": ARITHMETIC_INSTRUCTION,
    "<operator>.new": DATA_DECLARATION_INSTRUCTION,
    "<operator>.not": ARITHMETIC_INSTRUCTION,
    "<operator>.notEquals": COMPARE_INSTRUCTION,
    "<operator>.or": ARITHMETIC_INSTRUCTION,
    "<operator>.postDecrement": ARITHMETIC_INSTRUCTION,
    "<operator>.postIncrement": ARITHMETIC_INSTRUCTION,
    "<operator>.preDecrement": ARITHMETIC_INSTRUCTION,
    "<operator>.preIncrement": ARITHMETIC_INSTRUCTION,
    "<operator>.shiftLeft": ARITHMETIC_INSTRUCTION,
    "<operator>.sizeOf": ARITHMETIC_INSTRUCTION,
    "<operator>.subtraction": ARITHMETIC_INSTRUCTION,
    "<operators>.assignmentAnd": ARITHMETIC_INSTRUCTION,
    "<operators>.assignmentArithmeticShiftRight": ARITHMETIC_INSTRUCTION,
    "<operators>.assignmentOr": ARITHMETIC_INSTRUCTION,
    "<operators>.assignmentShiftLeft": ARITHMETIC_INSTRUCTION,
    "<operators>.assignmentXor": ARITHMETIC_INSTRUCTION,
    "ASSERT": COMPARE_INSTRUCTION,
    #"bool": DATA_DECLARATION_INSTRUCTION,
    "CASE": COMPARE_INSTRUCTION,
    "FIELD_IDENTIFIER": DATA_DECLARATION_INSTRUCTION,
    #"free": MOV_INSTRUCTION,
    "length": ARITHMETIC_INSTRUCTION,
    "memcmp": COMPARE_INSTRUCTION,
    "memcpy": TRANSFER_INSTRUCTION,
    "memset": TRANSFER_INSTRUCTION,
    "METHOD_RETURN": TERMINATION_INSTRUCTION,
    "METHOD": CALL_INSTRUCTION,
    "null": DATA_DECLARATION_INSTRUCTION,
    "RETURN": TERMINATION_INSTRUCTION,
    "strcasecmp": COMPARE_INSTRUCTION,
    "strcmp": COMPARE_INSTRUCTION,
    #"strcpy": TRANSFER_INSTRUCTION,
    "strlcpy": TRANSFER_INSTRUCTION,
    "strlen": ARITHMETIC_INSTRUCTION,
    "strncmp": COMPARE_INSTRUCTION,
    "strncpy": TRANSFER_INSTRUCTION,
    "strnlen": ARITHMETIC_INSTRUCTION
}
