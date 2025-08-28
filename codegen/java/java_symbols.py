from __future__ import annotations

from math_rep.expression_types import QualifiedName, MFunctionType, M_INT, M_BOOLEAN

JAVA_FRAME_NAME = '*java*'
JAVA_INTERNAL_FRAME_NAME = '*java-internal*'
JAVA_BUILTIN_FRAME_NAME = JAVA_INTERNAL_FRAME_NAME

JAVA_FRAME_PATH = (JAVA_FRAME_NAME,)
JAVA_INTERNAL_FRAME_PATH = (JAVA_INTERNAL_FRAME_NAME,)

MATH_LANG_JAVA_FRAME = tuple(reversed('java.lang.Math'.split('.')))

# FIXME!!! Need to have a solution to different method access, and their corresponding imports
#  equals() is java Object method, however,
#  the compareTo() is method from "Comparable" interface implemented by classes like "String"
#    i.e. "Comparable<String>"
#  need to find out how to express this differences in lexical path,
#  in this case we do not need to import any package,
OBJECT_EQUALS_QN = QualifiedName('equals', type=MFunctionType([], M_BOOLEAN, arity="?"), lexical_path=JAVA_FRAME_PATH)
COMPARABLE_COMPARE_TO_QN = QualifiedName('compareTo', type=MFunctionType([], M_INT, arity="?"),
                                         lexical_path=JAVA_FRAME_PATH)

MAX_LANG_MATH_QN = QualifiedName('max', lexical_path=MATH_LANG_JAVA_FRAME)
MIN_LANG_MATH_QN = QualifiedName('min', lexical_path=MATH_LANG_JAVA_FRAME)
JAVA_SET_CONTAINS_QN = QualifiedName('contains', lexical_path=('Set', 'util', 'java'))
JAVA_SET_OF_QN = QualifiedName('of', lexical_path=('Set', 'util', 'java'))
JAVA_LIST_CONTAINS_QN = QualifiedName('contains', lexical_path=('List', 'util', 'java'))
JAVA_LIST_OF_QN = QualifiedName('of', lexical_path=('List', 'util', 'java'))

JAVA_STREAM_QN = QualifiedName('stream', lexical_path=('Collection', 'util', 'java'))
JAVA_STREAM_FILTER_QN = QualifiedName('filter', lexical_path=('Stream', 'util', 'java'))
JAVA_STREAM_MAP_QN = QualifiedName('map', lexical_path=('Stream', 'util', 'java'))
JAVA_STREAM_FLATMAP_QN = QualifiedName('flatMap', lexical_path=('Stream', 'util', 'java'))
JAVA_STREAM_COLLECT_QN = QualifiedName('collect', lexical_path=('Stream', 'util', 'java'))
JAVA_STREAM_COLLECT_TO_LIST_QN = QualifiedName('toList', lexical_path=('Collectors', 'Stream', 'util', 'java'))
JAVA_STREAM_ITERATOR_NEXT_QU = QualifiedName('collect', lexical_path=tuple(reversed('java.util.Iterator'.split('.'))))

LAMBDA_PARAM_PATH = ('*lambda-param*',)


# FIXME: we need to add types
# Methods with type MFunctionType
# Classes with MClassType - the new one not the instance one
# TODO! encapsulate static call in a helper method
def static_method_to_class_transform(static_method: QualifiedName):
    return QualifiedName(static_method.lexical_path[0], lexical_path=static_method.lexical_path[1:])


class JavaFunctionModule:
    def __init__(self, func: QualifiedName):
        self.func = func

    def call(self):
        return QualifiedName(f'{self.func.lexical_path[0]}.{self.func.name}', lexical_path=self.func.lexical_path[1:])


# JAVA_LT_QN = QualifiedName('<', lexical_path=JAVA_INTERNAL_FRAME_PATH)
THIS = QualifiedName('this')
