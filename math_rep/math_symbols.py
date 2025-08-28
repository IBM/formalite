from typing import Sequence, AbstractSet, Optional, Union

from math_rep.constants import AND_SYMBOL, OR_SYMBOL, NOT_EQUALS_SYMBOL, LE_SYMBOL, GE_SYMBOL, FOR_ALL_SYMBOL, \
    IMPLIES_SYMBOL, NOT_SYMBOL
from math_rep.expression_types import QualifiedName, MFunctionType, M_BOOLEAN, M_NUMBER, MStreamType, M_ANY, M_INT, \
    MType, MArray, DeferredType
from math_rep.math_frame import MATH_FRAME_PATH

AND_QN = QualifiedName(AND_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
OR_QN = QualifiedName(OR_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
IMPLIES_QN = QualifiedName(IMPLIES_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
NOT_QN = QualifiedName(NOT_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)

LT_QN = QualifiedName('<', type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
LE_QN = QualifiedName(LE_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
GT_QN = QualifiedName('>', type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
GE_QN = QualifiedName(GE_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
EQ_QN = QualifiedName('=', type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
NEQ_QN = QualifiedName(NOT_EQUALS_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)

REVERSED_COMPARISON = {LT_QN: GT_QN,
                       GT_QN: LT_QN,
                       LE_QN: GE_QN,
                       GE_QN: LE_QN,
                       EQ_QN: NEQ_QN,
                       NEQ_QN: EQ_QN}

PLUS_SYMBOL = '+'
MINUS_SYMBOL = '-'
TIMES_SYMBOL = '*'
DIV_SYMBOL = '÷'
EXPON_SYMBOL = '^'  # replace with '↑'?  also other literal '^' symbols...
INT_DIV_SYMBOL = '/'
MOD_SYMBOL = '%'

PLUS_QN = QualifiedName(PLUS_SYMBOL, type=MFunctionType([], M_NUMBER, arity="?"), lexical_path=MATH_FRAME_PATH)
MINUS_QN = QualifiedName(MINUS_SYMBOL, type=MFunctionType([], M_NUMBER, arity="?"), lexical_path=MATH_FRAME_PATH)
TIMES_QN = QualifiedName(TIMES_SYMBOL, type=MFunctionType([], M_NUMBER, arity="?"), lexical_path=MATH_FRAME_PATH)
DIV_QN = QualifiedName(DIV_SYMBOL, type=MFunctionType([], M_NUMBER, arity="?"), lexical_path=MATH_FRAME_PATH)
INT_DIV_QN = QualifiedName(INT_DIV_SYMBOL, type=MFunctionType([M_INT, M_INT], M_INT), lexical_path=MATH_FRAME_PATH)
EXPON_QN = QualifiedName(EXPON_SYMBOL, type=MFunctionType([M_NUMBER, M_NUMBER], M_NUMBER),
                         lexical_path=MATH_FRAME_PATH)
ABS_QN = QualifiedName('abs', type=MFunctionType([M_NUMBER], M_NUMBER), lexical_path=MATH_FRAME_PATH)
MOD_QN = QualifiedName(MOD_SYMBOL, type=MFunctionType([M_INT, M_INT], M_INT), lexical_path=MATH_FRAME_PATH)
MATMUL_QN = QualifiedName('@', lexical_path=MATH_FRAME_PATH)
MIN_QN = QualifiedName('min', type=MFunctionType([], M_NUMBER, arity="?"), lexical_path=MATH_FRAME_PATH)
MAX_QN = QualifiedName('max', type=MFunctionType([], M_NUMBER, arity="?"), lexical_path=MATH_FRAME_PATH)
CEIL_QN = QualifiedName('ceil', type=MFunctionType([], M_NUMBER, arity="?"), lexical_path=MATH_FRAME_PATH)
IN_QN = QualifiedName('in', type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)
FOR_ALL_QN = QualifiedName(FOR_ALL_SYMBOL, type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=MATH_FRAME_PATH)

ZIP_QN = QualifiedName('zip', lexical_path=MATH_FRAME_PATH)
RANGE_QN = QualifiedName('range', lexical_path=MATH_FRAME_PATH)
STREAM_CONCATENATE_QN = QualifiedName('stream-concatenate',
                                      type=MFunctionType([MStreamType(M_ANY)], MStreamType(M_ANY), arity='*'),
                                      lexical_path=MATH_FRAME_PATH)
# like Stream.of
MAKE_STREAM_QN = QualifiedName('make-stream', type=MFunctionType([M_ANY], MStreamType(M_ANY), arity='*'),
                               lexical_path=MATH_FRAME_PATH)

MAKE_ARRAY_STR = 'make-array'


# constructor for array with a given set of elements
def make_array_qn(element_type: MType, *dims: Sequence[Optional[Union[MType, AbstractSet, range, DeferredType]]]):
    return QualifiedName(MAKE_ARRAY_STR, type=MFunctionType([element_type], MArray(element_type, *dims), arity='*'),
                         lexical_path=MATH_FRAME_PATH)


def is_make_array(qn: QualifiedName):
    return qn.name == MAKE_ARRAY_STR and qn.lexical_path == MATH_FRAME_PATH
