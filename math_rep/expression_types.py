from __future__ import annotations

import re
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from functools import total_ordering, reduce
from itertools import dropwhile, chain, repeat
from numbers import Number
from types import NoneType
from typing import Sequence, Set, TypeVar, Tuple, Union, AbstractSet, Optional, Iterator, ClassVar, Callable, List

from math_rep.math_frame import MATH_FRAME_PATH, MATH_FRAME_NAME
from meta_util.infrastructure import KeepMembersMixin
# TODO: implement subtyping relationships
from streams.stream_tools import successive_pairs_stream

JAVA_BOXED = {'byte': 'Byte', 'short': 'Short', 'int': 'Integer', 'long': 'Long', 'float': 'Float', 'double': 'Double',
              'boolean': 'Boolean', 'char': 'Character'}


class MType(ABC):
    @abstractmethod
    def for_python(self):
        pass

    def for_java(self):
        raise NotImplementedError()

    def for_java_boxed(self):
        base = self.for_java()
        return JAVA_BOXED.get(base.name, base.name)

    def __repr__(self):
        return self.__str__()

    # TODO: implement for all types and make abstract
    def for_opl(self):
        raise NotImplementedError()

    def def_for_opl(self, var_name: str):
        return f'{self.for_opl()} {var_name}'


class MAtomicType(MType, KeepMembersMixin):
    """
    An atomic type, such as Boolean, Number, String.

    These objects must be unique, equality is defined as object identity.
    """

    # TODO: use intern?
    def __init__(self, type_name: str, is_bounded=False, python_type=None, opl_type=None, java_type=None):
        self.type_name = type_name
        self.is_bounded = is_bounded
        self.python_type = python_type
        self.opl_type = opl_type
        self.java_type = java_type

    def as_bounded(self):
        return MAtomicType(self.type_name, True, self.python_type, self.opl_type, self.java_type)

    def for_python(self):
        return self.python_type

    def for_java(self):
        return QualifiedName(self.java_type, self, lexical_path=())

    def for_opl(self):
        return self.opl_type

    def __str__(self):
        return f'{self.type_name}'

    def __eq__(self, o: object) -> bool:
        return self is o
        # return isinstance(o, MAtomicType) and self.type_name == o.type_name

    def __hash__(self) -> int:
        return id(self)
        # return self.type_name.__hash__() * 5 + 1


@total_ordering
class MClassType(MType):
    def __init__(self, class_name: QualifiedName):
        self.class_name = class_name

    def for_python(self):
        return self.class_name.name

    def for_java(self):
        return self.class_name

    def for_opl(self):
        return self.class_name.name

    def __str__(self):
        return f'@{self.class_name}'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, MClassType) and self.class_name == o.class_name

    def __lt__(self, other):
        return self.class_name < other.class_name

    def __hash__(self) -> int:
        return self.class_name.__hash__() * 7 + 2


M_BOOLEAN = MAtomicType('Boolean', python_type='bool', opl_type='boolean', java_type='boolean')
M_NUMBER = MAtomicType('Number', python_type='float', opl_type='float', java_type='double')
M_FIXED_POINT = MAtomicType('Decimal', python_type='float', opl_type='float', java_type='*Decimal*')
M_INT = MAtomicType('Integer', python_type='int', opl_type='int', java_type='int')
M_STRING = MAtomicType('String', python_type='str', opl_type='string', java_type='String')
M_TYPE = MAtomicType('Type', python_type='type', java_type='')
M_NONE = MAtomicType('None', python_type='NoneType')
M_ANY = MAtomicType('Any', python_type='')
M_UNKNOWN = MAtomicType('???', python_type='')
M_BOTTOM = MAtomicType('⊥')

M_VOID = MAtomicType('Void', python_type=NoneType, java_type='void')

M_INT16 = MAtomicType('int16', python_type='int', opl_type='int', java_type='short')
M_INT32 = MAtomicType('int32', python_type='int', opl_type='int', java_type='int')
M_INT64 = MAtomicType('int64', python_type='int', opl_type='int', java_type='long')
M_FLOAT = MAtomicType('float', python_type='float', opl_type='float', java_type='float')
M_DOUBLE = MAtomicType('double', python_type='float', opl_type='float', java_type='double')

ATOMIC_PYTHON_TYPES = {t.for_python(): t for t in MAtomicType.members() if t.for_python()}

T = TypeVar('T')


def transitive_closure(chains: Set[Tuple[T, ...]]):
    """
    Compute the transitive closure of an anti-reflexive and antisymmetric relation
    :param chains: partial order, given as a set of chains of ascending elements (a, b, ...)
    :return: transitive closure, a set of pairs
    """
    pairs = set(chain.from_iterable(successive_pairs_stream(c) for c in chains))
    result = copy(pairs)
    while (add := {(a, c) for (a, b1) in pairs for (b2, c) in result if b1 == b2}) - result:
        result.update(add)
    return result


NUMERIC_TYPES = (M_BOOLEAN, M_INT16, M_INT32, M_INT64, M_INT, M_FIXED_POINT, M_FLOAT, M_DOUBLE, M_NUMBER)
NUMERIC_TYPES_INDEX = {t: ind for ind, t in enumerate(NUMERIC_TYPES)}
ATOMIC_SUBTYPES = transitive_closure({NUMERIC_TYPES})


class WithMembers(MType, ABC):
    def __init__(self, element_type):
        self.element_type = element_type

    def with_element_type(self, element_type: MType):
        return type(self)(element_type)


class MSetType(WithMembers):
    def __init__(self, element_type: MType):
        super().__init__(element_type)

    def for_python(self):
        return f'typing.AbstractSet[{self.element_type.for_python()}]'

    def for_opl(self):
        return f'{{{self.element_type.for_opl()}}}'

    def __str__(self):
        return f'Set[{self.element_type}]'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, MSetType) and self.element_type == o.element_type

    def __hash__(self) -> int:
        return self.element_type.__hash__()


class MStreamType(WithMembers):
    def __init__(self, element_type: MType):
        super().__init__(element_type)

    def for_python(self):
        return f'typing.Sequence[{self.element_type.for_python()}]'

    def __str__(self):
        return f'Stream[{self.element_type}]'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, MStreamType) and self.element_type == o.element_type

    def __hash__(self) -> int:
        return self.element_type.__hash__()

    def for_opl(self):
        # FIXME: implement
        raise NotImplementedError()


class MCollectionType(WithMembers):
    def __init__(self, element_type: MType):
        super().__init__(element_type)

    def for_python(self):
        return f'typing.Collection[{self.element_type.for_python()}]'

    def for_opl(self):
        # TODO: is this true in general?  It assumes that the subscripts will be provided elsewhere.
        return self.element_type.for_opl()

    def __str__(self):
        return f'Collection[{self.element_type}]'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, MCollectionType) and self.element_type == o.element_type

    def __hash__(self) -> int:
        return self.element_type.__hash__()


class MFunctionType(MType):
    def __init__(self, arg_types: Sequence[MType], result_type: MType, arity=None):
        """
        :param arg_types: sequence of argument types
        :param result_type: type of function result
        :param arity: None if arity is length of `arg_types`;
            '?' if unknown (in which case `arg_types` is ignored);
            '*' if last type can repeat 0 or more times
        """
        self.arg_types = tuple(arg_types)
        self.result_type = result_type
        self.arity = arity

    def arg_types_iter(self) -> Iterator[MType]:
        if self.arity == '*':
            return chain(self.arg_types, repeat(self.arg_types[-1]))
        return iter(self.arg_types)

    def matches(self, arg_types: Sequence[MType]) -> bool:
        if self.arity == '?':
            return True
        n_actual = len(arg_types)
        n_expected = len(self.arg_types)
        if n_actual < n_expected or self.arity != '*' and n_actual != n_expected:
            return False
        return all(is_subtype(actual_type, expected_type)
                   for actual_type, expected_type in zip(arg_types, self.arg_types_iter()))

    def with_result_type(self, result_type):
        return type(self)(self.arg_types, result_type, self.arity)

    def for_python(self):
        if self.arity is not None:
            params = '...'
        else:
            params = ', '.join(p.for_python() for p in self.arg_types)
        return f'typing.Callable[[{params}], {self.result_type.for_python() or "typing.Any"}]'

    def for_opl(self):
        raise Exception('OPL does not support functions')

    def __str__(self):
        if self.arity == '?':
            return f'?->{self.result_type}'
        return f'({", ".join(str(s) for s in self.arg_types)}{"*" if self.arity == "*" else ""})->{self.result_type}'

    def __eq__(self, o: object) -> bool:
        return (isinstance(o, MFunctionType)
                and self.arg_types == o.arg_types and self.result_type == o.result_type and self.arity == o.arity)

    def __hash__(self) -> int:
        return self.arg_types.__hash__() * 11 + self.result_type.__hash__() * 13 + self.arity.__hash__() * 17 + 3


class MUnionType(MType):
    def __init__(self, types: AbstractSet[MType]):
        # TODO: remove types that are subyptes of others
        self.types = frozenset({a for a in types if not any(b != a and is_subtype(a, b) for b in types)})

    def for_python(self):
        return f'typing.Union[{", ".join(t.for_python() for t in self.types)}]'

    def __new__(cls, types: Set[MType]):
        min_types = frozenset({a for a in types if not any(b != a and is_subtype(a, b) for b in types)})
        if len(min_types) == 1:
            return next(iter(min_types))
        if len(min_types) == 0:
            return M_ANY
        return super().__new__(cls)

    def __str__(self):
        return f'Union[{", ".join(str(s) for s in self.types)}]'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, MUnionType) and self.types == o.types

    def __hash__(self) -> int:
        return hash(self.types) * 23 + 4

    def for_opl(self):
        # FIXME: implement
        raise NotImplementedError()


class MTupleType(MType):
    def __init__(self, types: Sequence[MType]):
        self.types = types

    def __str__(self):
        return f'Tuple[{", ".join(str(s) for s in self.types)}]'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, MTupleType) and self.types == o.types

    def __hash__(self) -> int:
        return hash(self.types) * 101 + 6

    def for_python(self):
        # FIXME: implement
        raise NotImplementedError()

    def for_opl(self):
        # FIXME: implement
        raise NotImplementedError()


class DeferredType:
    pass


class MRange(MType):
    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop

    def get_value(self):
        return range(self.start, self.stop)

    def __eq__(self, other):
        return type(self) == type(other) and self.start == other.start and self.stop == other.stop

    def __hash__(self):
        return 23 * self.start - 17 * self.stop

    def __str__(self):
        return f'MRange({self.start}, {self.stop})'

    def for_python(self):
        return f'range({self.start}, {self.stop})'

    def for_opl(self):
        return f'{self.start}..{self.stop - 1}'


class MArray(WithMembers):
    # TODO: replace dims by MType's after supporting sets and ranges
    def __init__(self, element_type: MType, *dims: Sequence[Optional[Union[MType, AbstractSet, range, DeferredType]]]):
        super().__init__(element_type)
        self.dims = dims

    def with_dim(self, i: int, dim: Union[AbstractSet, range]):
        return type(self)(self.element_type, *[dim if j == i else d for j, d in enumerate(self.dims)])

    def __str__(self):
        return f'Array[{", ".join(str(d) for d in self.dims)}]:{self.element_type}'

    def __eq__(self, other):
        return isinstance(other, MArray) and self.element_type == other.element_type and self.dims == other.dims

    def __hash__(self):
        return 17 * hash(self.element_type) - 19 * hash(self.dims)

    def for_python(self):
        raise NotImplementedError('Not yet implemented')

    def for_opl(self):
        # FIXME!! create variables for the domains
        if any(dim is None for dim in self.dims):
            raise Exception(f'Incomplete dimensions for {self}')
        dim_strs = (f'{dim.start}..{dim.stop - 1}'
                    if isinstance(dim, range)
                    # FIXME!! fix for other types of dims
                    else f'{{{", ".join(repr(e) for e in sorted(dim))}}}'
                    for dim in self.dims)
        all_dims = ''.join(f'[{dim_s}]' for dim_s in dim_strs)
        return f'{self.element_type.for_opl()}{all_dims}'

    def def_for_opl(self, var_name):
        # FIXME!! create variables for the domains
        if any(dim is None for dim in self.dims):
            raise Exception(f'Incomplete dimensions for {self}')
        dim_strs = (f'{dim.start}..{dim.stop - 1}'
                    if isinstance(dim, range)
                    # FIXME!! fix for other types of dims
                    else f'{{{", ".join(repr(e) for e in sorted(dim))}}}'
                    for dim in self.dims)
        all_dims = ''.join(f'[{dim_s}]' for dim_s in dim_strs)
        return f'{self.element_type.for_opl()} {var_name}{all_dims}'


class MMappingType(MType):
    def __init__(self, key_type: MType, element_type: MType, total: bool):
        self.key_type = key_type
        self.element_type = element_type
        self.total = total

    def __str__(self):
        return f'{"TotalMapping" if self.total else "Mapping"}[{self.key_type}->{self.element_type}]'

    def __eq__(self, other):
        return (isinstance(other, MMappingType)
                and self.key_type == other.key_type
                and self.element_type == other.key_type
                and self.total == other.total)

    def __hash__(self):
        return 179 * hash(self.key_type) - 283 * hash(self.element_type)

    def for_python(self):
        if self.total:
            return f'optimization.TotalMapping[{self.key_type.for_python()}, {self.element_type.for_python()}]'
        return f'typing.Mapping[{self.key_type.for_python()}, {self.element_type.for_python()}]'

    def for_opl(self):
        raise NotImplementedError


class MRecordType(MType):
    def __init__(self, *fields: Tuple[str, MType]):
        self.fields = fields

    def __str__(self):
        fields_str = ', '.join(f'{ft[0]}: {str(ft[1])}' for ft in self.fields)
        return f'Record[{fields_str}]'

    def for_python(self):
        # FIXME!! need to create corresponding class; requires API change
        raise NotImplementedError(f'Record types not yet supported for Python')

    def for_opl(self):
        # FIXME!!!!! need to create corresponding struct; requires API change
        raise NotImplementedError(f'Record types not yet supported for OPL')
        pass


_DECIMAL_CLASSES: set[MClassType] = set()


def add_decimal_subtype(subtype: MClassType):
    """
    Add a class type as a subtype of M_FIXED_DECIMAL
    """
    assert isinstance(subtype, MClassType)
    global _DECIMAL_CLASSES
    _DECIMAL_CLASSES.add(subtype)


def is_subtype(a: MType, b: MType) -> bool:
    if a == b or b == M_ANY or a == M_BOTTOM:
        return True
    if issubclass(ta := type(a), MUnionType):
        return all(is_subtype(ma, b) for ma in a.types)
    if issubclass(tb := type(b), MUnionType):
        return any(is_subtype(a, mb) for mb in b.types)
    if issubclass(ta, MRange):
        if issubclass(tb, MRange):
            return a.start <= b.start and a.start + a.stop >= b.start + b.stop
        return is_subtype(M_INT, b)
    if issubclass(ta, MClassType):
        if a in _DECIMAL_CLASSES and is_subtype(M_FIXED_POINT, b):
            return True
        # FIXME: this may not work if the project isn't loaded, use inheritance hierarchy instead
        return (tb == ta
                and (ca := globals().get(a.class_name)) is not None
                and (cb := globals().get(b.class_name)) is not None
                and issubclass(ca, cb))
    if ta != tb:
        return False
    if ta == MAtomicType:
        return (a, b) in ATOMIC_SUBTYPES
    if issubclass(ta, MArray):
        return is_subtype(a.element_type, b.element_type) and all(
            ad == bd or isinstance(bd, DeferredType) for ad, bd in zip(a.dims, b.dims))
    if issubclass(ta, WithMembers):
        return is_subtype(a.element_type, b.element_type)
    if issubclass(ta, MFunctionType):
        if b.arity == '?':
            consistent_args = True
        elif a.arity == '?':
            consistent_args = False
        elif a.arity == '*':
            if b.arity != '*' or len(a.arg_types) > len(b.arg_types):
                consistent_args = False
            else:
                consistent_args = all(is_subtype(aarg, barg) for aarg, barg in zip(a.arg_types, b.arg_types))
        else:  # a.arity = None
            if len(b.arg_types) < len(a.arg_types):
                return False
            if len(b.arg_types) == len(a.arg_types) or len(b.arg_types) > len(a.arg_types) and b.arity == '*':
                consistent_args = all(is_subtype(aarg, barg) for aarg, barg in zip(a.arg_types, b.arg_types))
            else:
                consistent_args = False
        return consistent_args and is_subtype(a.result_type, b.result_type)
    if issubclass(ta, MMappingType):
        return (issubclass(tb, MMappingType)
                and is_subtype(b.key_type, a.key_type) and is_subtype(a.element_type, b.element_type)
                and a.total <= b.total)  # if subtype is total, supertype must also be total
    if issubclass(ta, MRecordType):
        return (issubclass(tb, MRecordType)
                and len(a.fields) == len(b.fields)
                and all(af[0] == bf[0] and issubclass(af[1], bf[1])
                        for af, bf in zip(a.fields, b.fields)))
    return False


def type_intersection(*types: MType) -> MType:
    return reduce(type_intersection2, types)


def type_intersection2(a: MType, b: MType) -> MType:
    if a == b:
        return a
    if is_subtype(a, b):
        return a
    if is_subtype(b, a):
        return b
    if isinstance(a, MUnionType) and isinstance(b, MUnionType):
        return MUnionType(a.types & b.types)
    if isinstance(a, MArray) and isinstance(b, MArray):
        if len(a.dims) != len(b.dims):
            return M_BOTTOM
        element_type = type_intersection(a.element_type, b.element_type)
        if element_type == M_BOTTOM:
            return M_BOTTOM
        dims = [ad if isinstance(bd, DeferredType) or ad == bd else bd if isinstance(ad, DeferredType) else 0
                for ad, bd in zip(a.dims, b.dims)]
        if any(d == 0 for d in dims):
            return M_BOTTOM
        return MArray(element_type, *dims)
    if isinstance(a, MFunctionType) and isinstance(b, MFunctionType):
        # compute most restrictive type
        if b.arity == '?':
            arity = b.arity
        elif a.arity == '?':
            arity = b.arity
        elif a.arity == '*':
            if b.arity == '*':
                arity = '*'
            elif len(a.arg_types) <= len(b.arg_types):
                arity = None  # b's arity fixed but within a's
            else:
                return M_BOTTOM  # inconsistent arities
        elif b.arity == '*':  # from here, a.arity is None
            if len(a.arg_types) >= len(b.arg_types):
                arity = None  # a's arity fixed but within b's
            else:
                return M_BOTTOM  # inconsistent arities
        elif len(a.arg_types) != len(b.arg_types):
            return M_BOTTOM
        else:
            arity = None  # both are fixed and equal length
        return MFunctionType(tuple(type_intersection(ai, bi) for ai, bi in zip(a.arg_types, b.arg_types)),
                             type_intersection(a.result_type, b.result_type),
                             arity=arity)
    # FIXME!! condition "a.key_type == b.key_type" should be removed, key type should be type_union of the keys,
    # add "a.key_type == b.key_type" to total
    if isinstance(a, MMappingType) and isinstance(b, MMappingType) and a.key_type == b.key_type:
        return MMappingType(type_intersection(a.key_type, b.key_type),
                            type_intersection(a.element_type, b.element_type),
                            a.total and b.total)
    if (isinstance(a, MRecordType) and isinstance(b, MRecordType)
            and len(a.fields) == len(b.fields)
            and all(af[0] == bf[0] for af, bf in zip(a.fields, b.fields))):
        return MRecordType(*((af[0], type_intersection(af[1], bf[1]))
                             for af, bf in zip(a.fields, b.fields)))
    if isinstance(a, MRange) and isinstance(b, MRange):
        return MRange(max(a.start, b.start), min(a.stop, b.stop))
    return M_BOTTOM


def type_of(value) -> MType:
    if isinstance(value, bool):
        return M_BOOLEAN
    if isinstance(value, Number):
        return M_NUMBER
    if isinstance(value, str):
        return M_STRING
    if isinstance(value, set):
        # N.B. True == 1, both will never appear in the same set!
        return MSetType(MUnionType({type_of(v) for v in value}))
    if value is None:
        return M_NONE
    return M_ANY


UNKNOWN_FRAME_NAME = '*unknowns*'


# QualifiedName is here because of mutual dependencies between names and types
@dataclass(frozen=True)
@total_ordering
class QualifiedName:
    name: Union[str, Tuple[str, ...]]
    type: MType = field(default=M_ANY, hash=False, compare=False)
    lexical_path: Sequence[str] = (UNKNOWN_FRAME_NAME,)
    lexical_scope_predicates: ClassVar[List[Callable[[QualifiedName], bool]]] = []

    def __eq__(self, other):
        return isinstance(other, QualifiedName) and self.name == other.name and self.lexical_path == other.lexical_path

    def __hash__(self):
        return 3 * hash(self.name) - 7 * hash(self.lexical_path)

    def __lt__(self, other):
        if not isinstance(other, QualifiedName):
            return NotImplemented
        return ((type(self.name).__name__, self.name, self.lexical_path) <
                (type(other.name).__name__, other.name, other.lexical_path))

    @classmethod
    def add_lexical_scope_predicate(cls, pred: Callable[[QualifiedName], bool]):
        cls.lexical_scope_predicates.append(pred)

    def is_lexically_scoped(self):
        return any(pred(self) for pred in self.lexical_scope_predicates)

    def to_c_identifier(self):
        return to_c_identifier(self.name)

    def with_path(self, path: Tuple[str, ...], override=False):
        if self.lexical_path == path:
            return self
        if not override and self.lexical_path and self.lexical_path != (UNKNOWN_FRAME_NAME,):
            raise Exception("Can't add a lexical path to an object that already has one")
        return QualifiedName(self.name, self.type, path)

    def with_type(self, new_type: MType) -> QualifiedName:
        return QualifiedName(self.name, new_type, lexical_path=self.lexical_path)

    def with_extended_path(self, *extensions) -> QualifiedName:
        return QualifiedName(self.name, self.type, lexical_path=(*reversed(extensions), *self.lexical_path))

    def with_extended_name(self, *components: str) -> QualifiedName:
        my_name = self.name if isinstance(self.name, (list, tuple)) else [self.name]
        return QualifiedName((*my_name, *components), self.type, self.lexical_path)

    def describe(self, full_path=False):
        if full_path:
            relevant_path = self.lexical_path
        else:
            relevant_path = list(dropwhile(lambda s: s.startswith('*'), self.lexical_path))
        path_str = '.'.join(reversed(relevant_path)) + '.' if relevant_path else ''
        type_str = '' if (t := self.type) in {M_UNKNOWN, M_ANY} or \
                         self.lexical_path in (MATH_FRAME_PATH,) else f':{str(t)}'
        return f'{path_str}{self.name}{type_str}'

    def __str__(self):
        return self.describe()

    def __repr__(self):
        return self.describe()


NON_ALPHANUMERIC_RE = re.compile('[^0-9a-zA-Z_]+')


def to_c_identifier(name: Union[str, Sequence[str]]) -> str:
    def fix_name_component(c):
        c = NON_ALPHANUMERIC_RE.sub('_', c)
        return c

    if isinstance(name, str):
        return fix_name_component(name)
    else:
        parts = [fix_name_component(str(e)) for e in name]
        return '_'.join(parts)


class NameTranslator:
    def translate(self, name: QualifiedName):
        # return unchanged as default
        return name


if __name__ == '__main__':
    # print(type_of({1, 2, 3}))
    # print(type_of({'a', 'b', 'c'}))
    # print(type_of({1, 'a', 'b', 'c'}))
    # print(type_of({2, 'a', True, 'c'}))
    # print(transitive_closure({(1, 2), (2, 3), (1, 4)}))

    # print(is_subtype(M_BOOLEAN, M_NUMBER))
    # print(is_subtype(M_BOOLEAN, M_ANY))
    # print(is_subtype(M_STRING, M_ANY))
    # print(not is_subtype(M_ANY, M_STRING))
    # print(not is_subtype(M_BOOLEAN, M_CELL))
    # print(is_subtype(MSetType(M_BOOLEAN), MSetType(M_NUMBER)))
    # print(is_subtype(MUnionType({M_BOOLEAN, MStreamType(M_CELL)}),
    #                  MUnionType({M_INT, MStreamType(MUnionType({M_CELL, M_RANGE}))})))
    # print(not is_subtype(MUnionType({M_INT, MStreamType(M_CELL)}),
    #                      MUnionType({M_BOOLEAN, MStreamType(MUnionType({M_CELL, M_RANGE}))})))
    # print(is_subtype(MFunctionType([M_BOOLEAN, M_INT], M_INT),
    #                  MFunctionType([M_NUMBER, M_ANY], M_NUMBER)))
    # # print(is_subtype(MClassType(QualifiedName('Tuple')), MClassType(QualifiedName('Sequence'))))
    # print(MUnionType({M_INT, M_BOOLEAN}) == M_INT)
    # print(type_intersection(MFunctionType((M_INT, M_BOOLEAN), M_NUMBER, arity='*'),
    #                         MFunctionType((M_NUMBER, M_INT), M_ANY)) ==
    #       MFunctionType((M_INT, M_BOOLEAN), M_NUMBER))
    print(is_subtype(MRange(1, 10), M_INT))
    print(type_intersection(MRange(1, 5), MRange(3, 7)))


def as_math_name(raw: str):
    return QualifiedName(raw, lexical_path=(MATH_FRAME_NAME,))


def is_math_name(qname: QualifiedName):
    return qname.lexical_path == MATH_FRAME_PATH
