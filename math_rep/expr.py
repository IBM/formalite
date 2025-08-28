from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from functools import wraps
from numbers import Number
from typing import Sequence, Union, Optional, Mapping, AbstractSet, MutableMapping, Tuple

import inflection
from c3linearize import linearize
from itertools import chain, islice, count, repeat, zip_longest, product

from codegen.abstract_rep import ComparisonExpr, AttributeAccess, VariableAccess, NumberExpr, StringExpr, SetExpr, \
    SetMembershipExpr, AggregateExpr, BetweenExpr, FunctionApplExpr, LogicalExpr, \
    PredicateApplExpr, NegatedExpr, Identifier, QuantifierExpr, TemporalExpr, BooleanExpr, ComprehensionContainerCode, \
    ComprehensionConditionCode, StreamExpr, AbstractFunctionDefinition, \
    AbstractClassDefinition, AbstractTypeDeclaration, AbstractNamedArg, CompilationUnit, SubscriptedExpr, \
    ConditionalExpr, RangeCode, SequenceExpr, LambdaExpressionExpr, CastExpr, NOT_OPERATOR, AND_OPERATOR, OR_OPERATOR, \
    PLUS_OPERATOR, TIMES_OPERATOR, NamedConstant
from codegen.parameters import ParameterDescriptor
from codegen.utils import Intern
from codegen.utils import decorate_method, Acceptor, disown, visitor_for
from math_rep.constants import ELEMENT_OF_SYMBOL, AND_SYMBOL, IMPLIES_SYMBOL, NOT_SYMBOL, OR_SYMBOL, LE_SYMBOL, \
    UC_SIGMA_SYMBOL, UC_PI_SYMBOL, INTERSECTION_SYMBOL, UNION_SYMBOL, NOT_EQUALS_SYMBOL, GE_SYMBOL, \
    NOT_ELEMENT_OF_SYMBOL, EPSILON_SYMBOL, FOR_ALL_SYMBOL, EXISTS_SYMBOL
from math_rep.expression_types import MFunctionType, M_NUMBER, M_ANY, MSetType, M_STRING, M_UNKNOWN, MType, \
    M_BOOLEAN, MStreamType, MUnionType, M_TYPE, M_BOTTOM, QualifiedName, \
    to_c_identifier, MArray, MCollectionType, M_INT, DeferredType, MRange, M_NONE, M_INT16, \
    M_INT32, M_INT64, MClassType, is_math_name
from math_rep.math_frame import MATH_FRAME_NAME
from math_rep.math_symbols import PLUS_QN, TIMES_QN, MINUS_QN, DIV_QN, MATMUL_QN, EQ_QN, NEQ_QN, LT_QN, LE_QN, \
    GT_QN, GE_QN, AND_QN, IMPLIES_QN, MIN_QN, MAX_QN
from rewriting.patterns import MetaPatternable, ClassPattern, find_recursive_matches

AGGREGATE_SYMBOL_MAP = {'+': UC_SIGMA_SYMBOL, '*': UC_PI_SYMBOL, 'SET': 'SET'}
AGGREGATE_TO_FUNCTION_QN_MAP = {'+': PLUS_OPERATOR, '*': TIMES_OPERATOR}

# Not used?
FUNCTION_SYMBOL_MAP = {'intersection': INTERSECTION_SYMBOL, 'union': UNION_SYMBOL, 'sum': '+', 'difference': '-',
                       'product': '*', 'quotient': '/', '%*': '*percent-of*',
                       # for Excel
                       '*': '*', '+': '+', '-': '-', 'max': 'max', 'if': 'if'}
FUNCTION_PRECEDENCE_MAP = {INTERSECTION_SYMBOL: 156, UNION_SYMBOL: 153, '+': 160, '-': 160, '*': 170, '/': 170,
                           '÷': 170, '%': 170, '^': 175}
ASSOCIATIVE_OPERATORS = {'+', '*', AND_SYMBOL, OR_SYMBOL}

TEMPORAL_BEFORE = '*t-before*'
TEMPORAL_AFTER = '*t-after*'
TEMPORAL_MEETS = '*t-meets*'
TEMPORAL_MEETS_INV = '*t-meets-inverse*'
TEMPORAL_DISJOINT = '*t-disjoint*'
TEMPORAL_OVERLAPS = '*t-overlaps*'

KNOWN_FUNCTIONS = {'+', '-', '*', '/', INTERSECTION_SYMBOL, UNION_SYMBOL, '*percent-of*', 'max', 'if',
                   '*concatenate-string*', '%', '^'}
KNOWN_COMPARISONS = {'=', NOT_EQUALS_SYMBOL, '<', GE_SYMBOL, '>', LE_SYMBOL, ELEMENT_OF_SYMBOL, NOT_ELEMENT_OF_SYMBOL}

LOGICAL_OP_BINDING_TABLE = {AND_SYMBOL: 110, OR_SYMBOL: 100, IMPLIES_SYMBOL: 90}
LOGICAL_OPS = LOGICAL_OP_BINDING_TABLE.keys()

SYMBOL_AS_TEXT = {OR_SYMBOL: 'or', AND_SYMBOL: 'and', IMPLIES_SYMBOL: 'implies'}

FUNCTION_TYPES = {'+': MFunctionType([M_NUMBER, M_NUMBER], M_NUMBER),
                  '/': MFunctionType([M_NUMBER, M_NUMBER], M_NUMBER),
                  # TODO: support different values for element-type (add lambda-types)
                  'set': MFunctionType([M_ANY], MSetType(M_ANY)),
                  # TODO: should specialize for M_PERIOD and sets (add lambda-types)
                  INTERSECTION_SYMBOL: MFunctionType([M_ANY, M_ANY], M_ANY)}
# ATTRIBUTE_TYPES = {'type': M_STRING, 'duration': M_PERIOD, 'length': M_NUMBER,
#                    'first-leg': MClassType('Leg'),
#                    'last-leg': MClassType('Leg'),
#                    'departure-station': MClassType('Station'),
#                    'arrival-station': MClassType('Station'),
#                    'period': M_PERIOD}
UNITS = {'%': '%'}


def normalize_key(obj):
    if isinstance(obj, FormalContent):
        return id(obj)
    if isinstance(obj, (list, tuple)):
        return tuple(normalize_key(e) for e in obj)
    if isinstance(obj, dict):
        return tuple((normalize_key(k), normalize_key(obj[k])) for k in sorted(obj.keys()))
    return obj


def intern_default_key(cls):
    def default_key(*args, **kwargs):
        return *(normalize_key(a) for a in args), normalize_key(kwargs)

    return default_key


def intern_key_no_kwargs(cls):
    def default_key(*args, **kwargs):
        return *(normalize_key(a) for a in args),

    return default_key


intern_expr = Intern(default_key=intern_default_key)


# def intern_expr(func=None, **kwargs):
#     if kwargs:
#         return lambda x: x
#     return func


# intern_expr.interned_classes = ()


def function_type(function_symbol: QualifiedName):
    default = function_symbol.type if isinstance(function_symbol.type, MFunctionType) \
        else MFunctionType((), M_ANY, arity='?')
    if is_math_name(function_symbol):
        return FUNCTION_TYPES.get(function_symbol.name, default)
    return default


def attribute_type(attribute_name, container_type):
    # FIXME!! this is for profiles only (and not used there); for Python, get from dataclass definition
    # t = ATTRIBUTE_TYPES.get(attribute_name.describe().lower())
    # if t is None or isinstance(t, MType):
    #     return t
    # return t.get(container_type, t.get('*'))
    return M_ANY


def normalize_unit(unit: str):
    # TODO: implement using mapping
    return UNITS.get(unit)


class ApplicationSpecificInfo(ABC):
    """
    Abstract superclass for application-specific information to be added to FormalContent objects.
    """

    @abstractmethod
    def get_type(self) -> MType:
        """
        Return the math type of the expression associated with this info
        """


def transfer_doc_and_type(method):
    """
    Transfer text to doc_string, and type to type, in generated code rep
    :param method:
    :return:
    """

    @wraps(method)
    def transferer(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        result.doc_string = self.text
        if not hasattr(result, 'type'):
            result.type = self.type
        result.prog_free_vars = self.free_vars
        return result

    return transferer


class FormalContent(Acceptor,
                    metaclass=decorate_method((transfer_doc_and_type, 'to_code_rep'),
                                              # supermetaclass=ABCMeta)):
                                              supermetaclass=MetaPatternable)):
    fresh_counter = defaultdict(int)

    @staticmethod
    def fresh_name(var: QualifiedName, indicator: str) -> QualifiedName:
        next_val = FormalContent.fresh_counter[indicator] + 1
        FormalContent.fresh_counter[indicator] = next_val
        return var.with_extended_path(f'*{indicator}{next_val}*')

    def __init__(self, free_vars: AbstractSet[QualifiedName] = frozenset()):
        self.free_vars = free_vars
        self.bound_vars = frozenset()
        self.text = ''
        self.appl_info: Optional[ApplicationSpecificInfo] = None

    own_binding = None
    has_operator = False

    def get_text(self):
        return self.text

    def set_text(self, text):
        if not self.text:
            self.text = text

    @abstractmethod
    def describe(self, parent_binding=None):
        """
        Return a printable description of this term
        :param parent_binding: binding strength of containing unit, used to add parentheses if necessary
        """

    def substitute(self, substitutions: Mapping[QualifiedName, 'FormalContent']) -> 'FormalContent':
        args = self.arguments()
        if not args:
            return self
        new_args = [arg.substitute(substitutions) if arg is not None else arg
                    for arg in args]
        return self.with_arguments(new_args)

    def skolemize(self, universals: Sequence[QualifiedName], var_factory: MathVariableFactory) -> 'FormalContent':
        args = self.arguments()
        new_args = [arg.skolemize(universals, var_factory) for arg in args]
        if any(na is not oa for na, oa in zip(new_args, args)):
            return self.with_arguments(new_args)
        return self

    # @abstractmethod
    def to_code_rep(self):
        raise Exception(f'{type(self).__name__}.to_code_rep() not implemented yet')

    def __str__(self):
        return self.describe()

    def __repr__(self):
        return self.describe()

    def parenthesize(self, description, parent_binding):
        if self.own_binding is not None and parent_binding is not None and parent_binding > self.own_binding:
            return f'({description})'
        return description

    def operator(self):
        return None

    def arguments(self) -> Iterable:
        return ()

    @abstractmethod
    def with_argument(self, index, arg: Term):
        raise NotImplementedError()

    def with_arguments(self, args: Sequence[Term]):
        # Default is to replace one by one, override with more efficient implementation where possible
        result = self
        for i, arg in enumerate(args):
            result = result.with_argument(i, arg)
            if not type(result) == type(self):
                return result
        return result

    def is_eq(self, other):
        return self is other or (type(self) == type(other) and self.operator() == other.operator()
                                 and all(sa is not None and sa.is_eq(oa)
                                         for sa, oa in zip_longest(self.arguments(), other.arguments())))

    def eq(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return Comparison(self, EQ_QN, other)
        raise Exception(f'Cannot compare {self.__class__.__name__} to {type(other)}')

    def __eq__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, str):
            other = StringTerm(other)
        if isinstance(other, Term):
            return Comparison(self, EQ_QN, other)
        if isinstance(other, FormalContent):
            return self.is_eq(other)
        return False

    def __ne__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return Comparison(self, NEQ_QN, other)
        if isinstance(other, FormalContent):
            return not self.is_eq(other)
        return True

    def ne(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return Comparison(self, NEQ_QN, other)
        raise Exception(f'Cannot compare {self.__class__.__name__} to {type(other)}')

    def __invert__(self):
        return LogicalOperator(NOT_OPERATOR, [self])


# # This use is discouraged; but is it really different?
# def __class_getitem__(cls, item) -> MathPatternElement:
#     return MathPattern(cls, item if isinstance(item, Tuple) else [item])

class Condition(FormalContent, ABC):
    # TODO: this is a problem since Negation doesn't have an operator, so Condition can't be used generically to
    #  include Negation
    has_operator = True

    def __init__(self):
        super().__init__()
        self.type = M_BOOLEAN

    # def describe(self, parent_binding=None):
    #     return self.parenthesize(f'Condition: {self.condition.describe(self.own_binding)}', parent_binding)

    # TODO: this applies to &, and will require non-intuitive parentheses; consider whether to support this
    def __and__(self, other):
        if isinstance(other, bool):
            if other:
                return self
            else:
                return Quantity(False, '*Boolean*')
        if isinstance(other, Condition):
            return LogicalOperator(AND_OPERATOR, [self, other])
        raise Exception(f'Cannot conjoin {self.__class__.__name__} with {type(other)}')

    def __rand__(self, other):
        if isinstance(other, bool):
            if other:
                return self
            else:
                return Quantity(False, '*Boolean*')
        if isinstance(other, Condition):
            return LogicalOperator(AND_OPERATOR, [other, self])
        raise Exception(f'Cannot conjoin {self.__class__.__name__} with {type(other)}')

    def __or__(self, other):
        if isinstance(other, bool):
            if not other:
                return self
            else:
                return Quantity(True, '*Boolean*')
        if isinstance(other, Condition):
            return LogicalOperator(OR_OPERATOR, [self, other])
        raise Exception(f'Cannot disjoin {self.__class__.__name__} with {type(other)}')

    def __ror__(self, other):
        if isinstance(other, bool):
            if not other:
                return self
            else:
                return Quantity(True, '*Boolean*')
        if isinstance(other, Condition):
            return LogicalOperator(OR_OPERATOR, [other, self])
        raise Exception(f'Cannot conjoin {self.__class__.__name__} with {type(other)}')


@intern_expr
class Negation(Condition):
    own_binding = 165  # 130
    has_operator = False

    def __init__(self, term: Condition):
        super().__init__()
        self.term = term
        self.free_vars = term.free_vars

    def to_code_rep(self):
        return NegatedExpr(self.term.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(NOT_SYMBOL + self.term.describe(self.own_binding), parent_binding)

    def arguments(self):
        return self.term,

    def with_argument(self, index, arg):
        if index != 0:
            raise IndexError('Index for Negate must be 0')
        return Negation(arg)


def with_element(elements, index, arg):
    if index not in range(len(elements)):
        raise IndexError(f'Index must be between 0 and {len(elements) - 1}')
    return [*elements[:index], arg, *elements[index + 1:]]


@intern_expr
class LogicalOperator(Condition):
    """
    A boolean logical operator.
    """

    @disown('kind')
    def __init__(self, kind, elements):
        try:
            if self.already_initialized:
                return
        except AttributeError:
            pass
        super().__init__()
        if isinstance(kind, QualifiedName):
            self.kind = kind
            self.own_binding = LOGICAL_OP_BINDING_TABLE.get(kind.name)
        else:
            assert kind in LOGICAL_OPS, f'Unknown logical operator {kind}'
            self.kind = QualifiedName(kind, lexical_path=(MATH_FRAME_NAME,))
            self.own_binding = LOGICAL_OP_BINDING_TABLE[kind]
        self.elements = elements
        self.free_vars = frozenset(chain.from_iterable(e.free_vars for e in self.elements))

    def __new__(cls, kind, elements):
        if kind == NOT_OPERATOR:
            assert len(elements) == 1, 'Only one argument allowed for negation'
            result = Negation(elements[0])
        elif kind == AND_OPERATOR:
            # Replace by False if any element is False
            if any(e.is_eq(FALSE_AS_QUANTITY) for e in elements):
                return FALSE_AS_QUANTITY
            # Remove True elements
            reduced_elements = [e for e in elements if e != TRUE_AS_QUANTITY]
            if not reduced_elements:
                return TRUE_AS_QUANTITY
            if len(reduced_elements) == 1:
                result = reduced_elements[0]
                result.already_initialized = True
                return result
            result = super().__new__(cls)
            if len(elements) != len(reduced_elements):
                result.__init__(kind, reduced_elements)
                result.already_initialized = True
        elif kind == OR_OPERATOR:
            # Replace by True if any element is True
            if any(e.is_eq(TRUE_AS_QUANTITY) for e in elements):
                return TRUE_AS_QUANTITY
            # Remove False elements
            reduced_elements = [e for e in elements if e != FALSE_AS_QUANTITY]
            if not reduced_elements:
                return FALSE_AS_QUANTITY
            if len(reduced_elements) == 1:
                result = reduced_elements[0]
                result.already_initialized = True
                return result
            result = super().__new__(cls)
            if len(elements) != len(reduced_elements):
                result.__init__(kind, reduced_elements)
                result.already_initialized = True
        else:
            result = super().__new__(cls)
        return result

    def to_code_rep(self):
        return LogicalExpr(self.kind, [e.to_code_rep() for e in self.elements])

    def describe(self, parent_binding=None):
        return self.parenthesize(f' {self.kind} '.join(c.describe(self.own_binding) for c in self.elements),
                                 parent_binding)

    def operator(self):
        return self.kind

    def arguments(self):
        return self.elements

    def with_argument(self, index, arg):
        return type(self)(self.kind, with_element(self.elements, index, arg))

    def with_arguments(self, args: Sequence[Term]):
        return type(self)(self.kind, args)


class LogicalOperatorAsExpression(LogicalOperator):
    """
    An indicator that this logical operator returns a non-boolean value and needs to be lifted across comparisons
    and function applications.
    """

    @disown('kind')
    def __init__(self, kind, elements):
        super().__init__(kind, elements)
        # override boolean type, this is an expression rather than a condition
        self.type = M_ANY


class PredicateAppl(Condition):
    has_operator = False

    def __init__(self, predicate, args):
        super().__init__()
        self.predicate = predicate
        self.args = args
        self.free_vars = frozenset(chain.from_iterable(e.free_vars for e in self.args))

    def to_code_rep(self):
        return PredicateApplExpr(self.predicate.to_code_rep(), [a.to_code_rep() for a in self.args])

    def describe(self, parent_binding=None):
        return self.parenthesize(self.predicate.describe(self.own_binding) +
                                 '(' + ', '.join(arg.describe() for arg in self.args) + ')',
                                 parent_binding)

    def arguments(self):
        return [self.predicate] + self.args

    def with_argument(self, index, arg):
        if index == 0:
            return PredicateAppl(arg, self.args)
        return PredicateAppl(self.predicate, with_element(self.args, index - 1, arg))

    def with_arguments(self, args: Sequence[Term]):
        return type(self)(self.predicate, args)


class Term(FormalContent, ABC):
    has_operator = True

    @disown('type')
    def __init__(self, type: MType):
        super().__init__()
        self.type = type or M_ANY

    def get_type(self):
        return self.type

    def __add__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return FunctionApplication(PLUS_QN, (self, other))
        raise Exception(f'Cannot add {self.__class__.__name__} to {type(other)}')

    def __radd__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        # TODO: this should already be covered by __add__, remove
        if isinstance(other, Term):
            return FunctionApplication(PLUS_QN, (other, self))
        raise Exception(f'Cannot add {type(other)} to {self.__class__.__name__}')

    def __pos__(self):
        return self

    def __sub__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return FunctionApplication(MINUS_QN, (self, other))
        raise Exception(f'Cannot subtract {type(other)} from {self.__class__.__name__}')

    def __rsub__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        # TODO: this should already be covered by __sub__, remove
        if isinstance(other, Term):
            return FunctionApplication(MINUS_QN, (other, self))
        raise Exception(f'Cannot subtract {self.__class__.__name__} from {type(other)}')

    def __neg__(self):
        return FunctionApplication(MINUS_QN, (self,))

    def __mul__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return FunctionApplication(TIMES_QN, (self, other))
        raise Exception(f'Cannot multiply {self.__class__.__name__} by {type(other)}')

    def __rmul__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        # TODO: this should already be covered by __mul__, remove
        if isinstance(other, Term):
            return FunctionApplication(TIMES_QN, (other, self))
        raise Exception(f'Cannot multiply {type(other)} by {self.__class__.__name__}')

    def __truediv__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return FunctionApplication(DIV_QN, (self, other))
        raise Exception(f'Cannot divide {self.__class__.__name__} by {type(other)}')

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        # TODO: this should already be covered by __truediv__, remove
        if isinstance(other, Term):
            return FunctionApplication(DIV_QN, (other, self))
        raise Exception(f'Cannot divide {type(other)} by {self.__class__.__name__}')

    def __matmul__(self, other):
        if isinstance(other, Term):
            return FunctionApplication(MATMUL_QN, (self, other))
        raise Exception(f'Cannot matrix-multiply {self.__class__.__name__} by {type(other)}')

    def __rmatmul__(self, other):
        if isinstance(other, Term):
            return FunctionApplication(MATMUL_QN, (other, self))
        raise Exception(f'Cannot matrix-multiply {type(other)} by {self.__class__.__name__}')

    def __lt__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return Comparison(self, LT_QN, other)
        raise Exception(f'Cannot compare {self.__class__.__name__} to {type(other)}')

    def __le__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return Comparison(self, LE_QN, other)
        raise Exception(f'Cannot compare {self.__class__.__name__} to {type(other)}')

    def __gt__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return Comparison(self, GT_QN, other)
        raise Exception(f'Cannot compare {self.__class__.__name__} to {type(other)}')

    def __ge__(self, other):
        if isinstance(other, Number):
            other = Quantity(other)
        if isinstance(other, Term):
            return Comparison(self, GE_QN, other)
        raise Exception(f'Cannot compare {self.__class__.__name__} to {type(other)}')


class SetMembership(Condition):
    own_binding = 150
    has_operator = False

    def __init__(self, element: FormalContent, container: Term):
        super().__init__()
        self.element = element
        self.container = container
        self.free_vars = element.free_vars | container.free_vars

    def to_code_rep(self):
        return SetMembershipExpr(self.element.to_code_rep(), self.container.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(f'{self.element.describe(self.own_binding)} {ELEMENT_OF_SYMBOL} '
                                 f'{self.container.describe(self.own_binding)}',
                                 parent_binding)

    def arguments(self):
        return self.element, self.container

    def with_argument(self, index, arg):
        if index == 0:
            return SetMembership(arg, self.container)
        if index == 1:
            return SetMembership(self.element, arg)
        raise IndexError('Index for SetMembership must be 0 or 1')

    def __new__(cls, expr, container):
        if isinstance(expr, LogicalCombination):
            elements = [SetMembership(e, container) for e in expr.elements]
            for sm, exp in zip(elements, expr.elements):
                sm.set_text(f'{exp.get_text()} in {container.get_text()}')
            result = LogicalOperator(expr.combiner, elements)
        elif isinstance(container, LogicalCombination):
            elements = [SetMembership(expr, c) for c in container.elements]
            for sm, cont in zip(elements, container.elements):
                sm.set_text(f'{expr.get_text()} in {cont.get_text()}')
            result = LogicalOperator(container.combiner, elements)
        else:
            result = super().__new__(cls)
        return result


@intern_expr
class Comparison(Condition):
    own_binding = 140

    @disown('op')
    def __init__(self, lhs: Term, op, rhs: Term):
        super().__init__()
        # TODO: op should always be a QualifiedName
        if isinstance(op, str):
            assert op in KNOWN_COMPARISONS, f'Unknown comparison operator {op}'
            op = QualifiedName(op, lexical_path=(MATH_FRAME_NAME,))
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.free_vars = lhs.free_vars | rhs.free_vars

    def __bool__(self):
        if self.op.name == '=':
            return self.lhs.is_eq(self.rhs)
        if self.op.name == NOT_EQUALS_SYMBOL:
            return not self.lhs.is_eq(self.rhs)
        raise Exception("Can't compare terms with inequalities")

    def operator(self):
        return self.op

    def arguments(self):
        return self.lhs, self.rhs

    def with_argument(self, index, arg):
        if index == 0:
            result = Comparison(arg, self.op, self.rhs)
        elif index == 1:
            result = Comparison(self.lhs, self.op, arg)
        else:
            raise IndexError('Index for Comparison must be 0 or 1')
        try:
            result.is_definition = self.is_definition
        except AttributeError:
            pass
        return result

    def __new__(cls, lhs, op, rhs):
        if isinstance(lhs, LogicalCombination):
            elements = [Comparison(e, op, rhs) for e in lhs.elements]
            for comp, exp in zip(elements, lhs.elements):
                comp.set_text(f'{exp.get_text()} {op} {rhs.get_text()}')
            result = LogicalOperator(lhs.combiner, elements)
        elif isinstance(rhs, LogicalCombination):
            elements = [Comparison(lhs, op, e) for e in rhs.elements]
            for comp, exp in zip(elements, rhs.elements):
                comp.set_text(f'{lhs.get_text()} {op} {exp.get_text()}')
            result = LogicalOperator(rhs.combiner, elements)
        else:
            result = super().__new__(cls)
        return result

    def to_code_rep(self):
        return ComparisonExpr(self.lhs.to_code_rep(), self.op, self.rhs.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(
            f'{self.lhs.describe(self.own_binding)} {self.op} {self.rhs.describe(self.own_binding)}',
            parent_binding)


@intern_expr
class TemporalOrder(Condition):
    @disown('op')
    def __init__(self, lhs: Term, op, rhs: Term, container: Term):
        super().__init__()
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.container = container
        self.free_vars = lhs.free_vars | rhs.free_vars | container.free_vars

    def to_code_rep(self):
        return TemporalExpr(self.lhs.to_code_rep(), self.op, self.rhs.to_code_rep(), self.container.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(
            f'{self.op}({self.lhs.describe(self.own_binding)}, {self.rhs.describe(self.own_binding)}, '
            f'{self.container.describe(self.own_binding)})',
            parent_binding)

    def operator(self):
        return self.op

    def arguments(self):
        return self.lhs, self.rhs, self.container

    def with_argument(self, index, arg):
        if index == 0:
            return TemporalOrder(arg, self.op, self.rhs, self.container)
        if index == 1:
            return TemporalOrder(self.lhs, self.op, arg, self.container)
        if index == 2:
            return TemporalOrder(self.lhs, self.op, self.rhs, arg)
        raise IndexError('Index for Comparison must be between 0 and 2')


@intern_expr
class Attribute(Term):
    own_binding = None  # 190 # for dot notation
    has_operator = False

    @disown('type')
    def __init__(self, attribute: Term, container: Term, type=M_ANY):
        my_type = type or attribute_type(attribute, container.get_type())
        super().__init__(my_type)
        self.attribute = attribute
        self.container = container
        self.free_vars = attribute.free_vars | container.free_vars

    def to_code_rep(self):
        return AttributeAccess(self.attribute.to_code_rep(), self.container.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(
            # self.container.describe(self.own_binding) + '.' + self.attribute.describe(self.own_binding),
            f'{self.attribute.describe(self.own_binding)}({self.container.describe(self.own_binding)})',
            parent_binding)

    def __new__(cls, attribute, container, type=M_ANY):
        if isinstance(attribute, LogicalCombination):
            elements = [Attribute(a, container) for a in attribute.elements]
            for attr, exp in zip(elements, attribute.elements):
                attr.set_text(f'{exp.get_text()} of {container.get_text()}')
            result = LogicalCombination(attribute.combiner, elements)
        elif isinstance(container, LogicalCombination):
            elements = [Attribute(attribute, c) for c in container.elements]
            for attr, exp in zip(elements, container.elements):
                attr.set_text(f'{attribute.get_text()} of {exp.get_text()}')
            result = LogicalCombination(container.combiner, elements)
        else:
            result = super().__new__(cls)
        return result

    def arguments(self):
        return self.attribute, self.container

    def with_argument(self, index, arg):
        if index == 0:
            return Attribute(arg, self.container)
        if index == 1:
            return Attribute(self.attribute, arg)
        raise IndexError('Index for Attribute must be 0 or 1')


@intern_expr
class Atom(Term):
    @disown('type_name', 'words', 'article', 'lexical_path')
    def __init__(self, type_name, words, article=None, lexical_path=()):
        super().__init__(type_name)
        self.words = words
        self.article = article
        self.lexical_path = lexical_path

    def to_code_rep(self):
        return Identifier(QualifiedName(self.words, lexical_path=self.lexical_path))

    def to_c_identifier(self):
        return to_c_identifier(self.words)

    def describe(self, parent_binding=None):
        # article_text = f'[{self.article}]' if self.article else ''
        # result = article_text + '-'.join(self.words)
        result = '-'.join(self.words)
        return self.parenthesize(result, parent_binding)

    def arguments(self):
        return ()

    def with_argument(self, index, arg):
        raise IndexError('Cannot subscript Atom')

    def is_eq(self, other):
        return type(self) == type(other) and self.words == other.words and self.lexical_path == other.lexical_path


@intern_expr
class Predicate(Atom):
    @disown('words', 'article', 'negated', 'lexical_path')
    def __init__(self, words, article=None, negated=False, lexical_path=()):
        # TODO: add mapping of predicates to more precise types
        super().__init__(MFunctionType((), M_BOOLEAN, arity='?'), words, article=article)
        self.negated = negated
        self.lexical_path = lexical_path

    @staticmethod
    def from_atom(atom, negated=False):
        return Predicate(atom.words, atom.article, negated, lexical_path=atom.lexical_path)

    def to_positive(self):
        return Predicate(self.words, article=self.article, negated=False, lexical_path=self.lexical_path)

    def describe(self, parent_binding=None):
        result = ('NEG-' if self.negated else '') + super().describe(parent_binding)
        return self.parenthesize(result, parent_binding)

    def is_eq(self, other):
        return super().is_eq(other) and self.negated == other.negated


# TODO: remove this (replaced by rules)
@intern_expr
class LogicalCombination(Term):
    """
    A combination of expressions using logical operators; the combination needs to be lifted to the boolean level.
    """

    @disown('combiner')
    def __init__(self, combiner, elements: Sequence[Term]):
        super().__init__(MFunctionType((M_BOOLEAN, M_BOOLEAN), M_BOOLEAN))
        if isinstance(combiner, QualifiedName):
            self.combiner = combiner
            self.own_binding = LOGICAL_OP_BINDING_TABLE.get(combiner.name)
        else:
            assert combiner in LOGICAL_OPS, f'Unknown logical combination {combiner}'
            self.combiner = QualifiedName(combiner, lexical_path=(MATH_FRAME_NAME,))
            self.own_binding = LOGICAL_OP_BINDING_TABLE[combiner]
        self.elements = elements
        self.free_vars = frozenset(chain.from_iterable(e.free_vars for e in self.elements))

    def describe(self, parent_binding=None):
        return self.parenthesize(f' {self.combiner} '.join(p.describe(self.own_binding) for p in self.elements),
                                 parent_binding)

    def operator(self):
        return self.combiner

    def arguments(self):
        return self.elements

    def with_argument(self, index, arg):
        return LogicalCombination(self.combiner, with_element(self.elements, index, arg))

    def with_arguments(self, args: Sequence[Term]):
        return type(self)(self.combiner, args)


@intern_expr
class Package(FormalContent):
    """
    A package or module, with a hierarchical name
    """

    @disown('path')
    def __init__(self, path: Sequence[str]):
        super().__init__()
        self.path = path

    def describe(self, parent_binding=None):
        name = '.'.join(self.path)
        return f'Package {name}'

    def with_argument(self, index, arg):
        return self


@Intern(default_key=intern_key_no_kwargs)
class MathVariable(Term):
    @disown('name', 'override_type')
    def __init__(self, name: QualifiedName, override_type: MType = None):
        super().__init__(override_type or name.type)
        self.name = name
        self.set_text(name.name)
        self.free_vars = frozenset([name])

    def with_type(self, type: MType):
        if type == self.type:
            return self
        return MathVariable(self.name, override_type=type)

    def to_code_rep(self):
        return VariableAccess(self.name)

    def describe(self, parent_binding=None):
        return self.parenthesize(f'${self.name.name}', parent_binding)

    def operator(self):
        return self.name

    def arguments(self):
        return ()

    def with_argument(self, index, arg):
        raise ValueError('Cannot subscript MathVariable')

    def substitute(self, substitutions: Mapping[QualifiedName, Term]) -> Term:
        # if type(self) == MathVariable:
        #     # Do this only for primitive MathVariable's
        return substitutions.get(self.name, self)
        # return super().substitute(substitutions)

    def is_eq(self, other):
        return type(other) == MathVariable and self.name == other.name

    def defined_as(self, expr: Union[Term, Number]) -> Comparison:
        if isinstance(expr, Number):
            expr = Quantity(expr)
        result = Comparison(self, '=', expr)
        result.is_definition = True
        return result

    def is_output(self):
        return hasattr(self, 'is_output_var')

    def var_name(self):
        # TODO: replace this method by self.var when MathVariableArray.name renamed to var
        return self.name


@intern_expr
class IndexedMathVariable(Term):
    """
    A MathVariable that is part of a MathVariableArray, with the index(es) into the array
    """
    has_operator = False

    def __init__(self, owner: MathVariableArray, *indexes: Term):
        try:
            e_type = owner.type.element_type
        except AttributeError:
            e_type = M_ANY
        super().__init__(e_type)
        # ??? owner.name.with_extended_name([str(idx) for idx in indexes])
        self.owner = owner
        self.indexes = indexes

    def to_code_rep(self):
        return SubscriptedExpr(self.owner.to_code_rep(), [index.to_code_rep() for index in self.indexes])

    def describe(self, parent_binding=None):
        index_string = ', '.join(str(index) for index in self.indexes)
        return self.parenthesize(f'{self.owner}[{index_string}]', parent_binding)

    def arguments(self):
        return self.owner, *self.indexes

    def with_argument(self, index, arg):
        if index == 0:
            return type(self)(arg, *self.indexes)
        return type(self)(self.owner, *[arg if index == i + 1 else idx for i, idx in enumerate(self.indexes)])

    def is_eq(self, other):
        return type(other) == IndexedMathVariable and self.owner.is_eq(
            other.owner) and self.indexes == other.indexes

    def defined_as(self, expr: Term) -> Comparison:
        result = Comparison(self, '=', expr)
        result.is_definition = True
        return result

    def is_output(self):
        var = self
        while isinstance(var, IndexedMathVariable):
            if hasattr(var, 'is_output_var'):
                return True
            var = var.owner
        return hasattr(var, 'is_output_var')


@intern_expr
class MathVariableArray(Term):
    """
    A one-dimensional array of MathVariable objects.  This cannot be directly translated into a program, but needs
    to be converted into something else by rewrite rules.
    """
    has_operator = False

    @disown('name')
    # FIXME! use TypeExpr with a new MType for a set of values to support string values
    def __init__(self, name: QualifiedName, *dims: Union[DomainDim, RangeExpr, TypeExpr]):
        super().__init__(MArray(name.type, *[d.as_range() if isinstance(d, RangeExpr)
                                             else d.mtype if isinstance(d, TypeExpr)
        else d
                                             for d in dims]))
        self.name = name
        self.dims = dims
        self.instances: MutableMapping[Tuple[Union[int, str], ...], IndexedMathVariable] = {}

    def to_code_rep(self):
        return VariableAccess(self.name)

    def get_var(self, *indexes: Term):
        # if all(isinstance(index, Quantity) and isinstance(index.value, (str, int)) and index.value in dim
        #        for index, dim in zip(indexes, self.dims)):
        return IndexedMathVariable(self, *indexes)

    def arguments(self):
        return ()

    def describe(self, parent_binding=None):
        return f'$${self.name}'
        # return self.parenthesize(f'{self.name}[{", ".join(repr(dim) for dim in self.dims)}]', parent_binding)

    def with_argument(self, index, arg):
        raise ValueError('Cannot replace arguments for MathVariableArray')

    def substitute(self, substitutions: Mapping[QualifiedName, FormalContent]) -> MathVariableArray:
        # NOTE: substitution of MVA's not currently supported!
        dims = [dim.substitute(substitutions) for dim in self.dims]
        if all(nd is d for nd, d in zip(dims, self.dims)):
            return self
        return MathVariableArray(self.name, *dims)

    def is_output(self):
        return hasattr(self, 'is_output_var')

    def is_eq(self, other):
        return type(other) == MathVariableArray and self.name == other.name and self.dims == other.dims

    # def __hash__(self):
    #     return hash(self.name) + sum(w * hash(dim) for w, dim in zip(count(3, 2), self.dims))

    def _index_set(self, index: int, item) -> Tuple[Term, ...]:
        dim = self.dims[index]
        if isinstance(item, Quantity) and item.type is M_INT:
            item = item.value
        if isinstance(item, int):
            if not isinstance(dim, RangeExpr):
                return Quantity(item),
            lower = dim.start
            upper = dim.stop
            orig_item = item
            if item < 0:
                item = upper + item
            if not lower <= item < upper:
                item_str = str(item) if item == orig_item else f'{orig_item} (i.e., {item})'
                raise IndexError(f'Index {item_str} out of range {lower}..{upper - 1} for '
                                 f'{inflection.ordinalize(index + 1)} index of {repr(self)}')
            return Quantity(item),
        if isinstance(item, slice):
            if not isinstance(dim, RangeExpr):
                raise Exception(f'Can only slice a range: {item} in {dim}')
            lower = dim.start
            upper = dim.stop
            return tuple(Quantity(index)
                         for index in islice(count(lower), *item.indices(upper - lower)))
        if isinstance(item, (range, RangeExpr)):
            if isinstance(dim, RangeExpr):
                lower = dim.start
                upper = dim.stop
                if item.start < lower:
                    raise IndexError(f'Index {item.start} out of range {lower}..{upper - 1} for '
                                     f'{inflection.ordinalize(index + 1)} index of {repr(self)}')
                if item.stop > upper:
                    raise IndexError(f'Index {item.stop - 1} out of range {lower}..{upper - 1} for '
                                     f'{inflection.ordinalize(index + 1)} index of {repr(self)}')
            return tuple(Quantity(index) for index in item)
        if isinstance(item, Quantity) and item.type is M_STRING and isinstance(dim, RangeExpr):
            raise IndexError(f'String index {item} for numeric dimension {dim}')
        return item,

    def __getitem__(self, item):
        """
        Return a tuple of IndexedMathVariable terms if `item` is a RangeExpr or a slice, and an indexed item otherwise.
        """
        if isinstance(item, tuple):
            item_set = product(*(self._index_set(index, x) for index, x in enumerate(item)))
        else:
            item_set = [self._index_set(0, item)]
        result = []
        for items in item_set:
            if all(isinstance(x, Quantity) for x in items):
                key = tuple(x.value for x in items)
                imv = self.instances.get(key)
                if imv is None:
                    imv = IndexedMathVariable(self, *items)
                    self.instances[key] = imv
            else:
                imv = IndexedMathVariable(self, *items)
            result.append(imv)
        return result[0] if len(result) == 1 else result

    def var_name(self):
        # TODO: replace this method by self.var when MathVariableArray.name renamed to var
        return self.name


@intern_expr(key=lambda value, unit=None: (str(type(value)), value, unit))
class Quantity(Term):
    # Treating unit as operator
    has_operator = True

    @disown('value', 'unit')
    def __init__(self, value, unit=None):
        if isinstance(value, bool):
            qtype = M_BOOLEAN
            if unit is None:
                unit = '*Boolean*'
        elif isinstance(value, str):
            qtype = M_STRING
            if unit is None:
                unit = '*String*'
        elif isinstance(value, int):
            av = abs(value)
            if av < pow(2, 15):
                qtype = M_INT16
            elif av < pow(2, 31):
                qtype = M_INT32
            elif av < pow(2, 63):
                qtype = M_INT64
            else:
                qtype = M_INT
        elif isinstance(value, float):
            qtype = M_NUMBER
        elif value is None:
            qtype = M_NONE
        else:
            qtype = normalize_unit(unit) if unit else M_NUMBER
        super().__init__(qtype)
        self.value = value
        self.unit = unit

    def is_eq(self, other):
        return isinstance(other, Quantity) and self.value == other.value and self.unit == other.unit

    def to_code_rep(self):
        # FIXME: convert units as needed
        if self.unit == '*Boolean*':
            return BooleanExpr(self.value)
        elif self.unit == '*String*':
            return StringExpr(self.value)
        return NumberExpr(self.value)

    def describe(self, parent_binding=None):
        return self.parenthesize(str(self.value) + (self.unit or ''), parent_binding)

    def operator(self):
        return self.unit

    def with_argument(self, index, arg):
        raise IndexError('Quantity has no arguments')


TRUE_AS_QUANTITY = Quantity(True, '*Boolean*')
FALSE_AS_QUANTITY = Quantity(False, '*Boolean*')
ZERO_AS_QUANTITY = Quantity(0)
ONE_AS_QUANTITY = Quantity(1)


@intern_expr
class StringTerm(Term):
    # Treating contents as operator
    has_operator = True

    @disown('contents')
    def __init__(self, contents):
        super().__init__(M_STRING)
        self.contents = contents

    def describe(self, parent_binding=None):
        return self.parenthesize(f'"{self.contents}"', parent_binding)

    def to_code_rep(self):
        return StringExpr(self.contents)

    def operator(self):
        return self.contents

    def with_argument(self, index, arg):
        raise IndexError('StringTerm has no arguments')


@intern_expr
class IFTE(Term):
    own_binding = 50
    has_operator = False

    def __init__(self, cond: Condition, pos: Term, neg: Term):
        super().__init__(pos.type if pos.type == neg.type else MUnionType({pos.type, neg.type}))
        self.cond = cond
        self.pos = pos
        self.neg = neg
        self.free_vars = cond.free_vars | pos.free_vars | neg.free_vars

    def describe(self, parent_binding=None):
        return self.parenthesize(f'{self.cond.describe()} ? {self.pos.describe()} : {self.neg.describe()}',
                                 parent_binding)

    def to_code_rep(self):
        return ConditionalExpr(self.cond.to_code_rep(), self.pos.to_code_rep(), self.neg.to_code_rep())

    def arguments(self):
        return self.cond, self.pos, self.neg

    def with_argument(self, index, arg):
        return IFTE(*with_element(self.arguments(), index, arg))


@intern_expr
class Subscripted(Term):
    # FIXME: distinguish between arrays and dictionaries -- requires Python type analysis for Python sources
    own_binding = 180
    has_operator = False

    def __init__(self, obj: Term, subscripts: Sequence[Term]):
        # FIXME: keep track of types (add array/dict types)
        my_type = M_ANY
        super().__init__(my_type)
        self.obj = obj
        self.subscripts = subscripts
        self.free_vars = obj.free_vars | frozenset(chain.from_iterable(e.free_vars for e in self.subscripts))

    def describe(self, parent_binding=None):
        subs = ', '.join(s.describe() for s in self.subscripts)
        return self.parenthesize(f'{self.obj.describe()}[{subs}]', parent_binding)

    def to_code_rep(self):
        return SubscriptedExpr(self.obj.to_code_rep(), [s.to_code_rep() for s in self.subscripts])

    def arguments(self):
        return [self.obj] + list(self.subscripts)

    def with_argument(self, index, arg):
        if index == 0:
            return Subscripted(arg, self.subscripts)
        return Subscripted(self.obj, with_element(self.subscripts, index - 1, arg))


class ComprehensionElement(FormalContent, ABC):
    def __init__(self):
        super().__init__()
        self.type = MStreamType(M_ANY)
        self.text = ''

    def arguments(self):
        raise ValueError('Cannot subscript ComprehensionElement directly')


@intern_expr
class ComprehensionContainer(ComprehensionElement):
    has_operator = False

    @disown('vars')
    def __init__(self, vars: Sequence[QualifiedName], container: Term, rest: Optional[ComprehensionElement] = None):
        super().__init__()
        self.vars = vars
        self.container = container
        self.rest = rest
        sub_vars = container.free_vars
        bound_vars = frozenset(vars)
        if rest is not None:
            sub_vars = sub_vars | rest.free_vars
            bound_vars = bound_vars | rest.bound_vars
        self.free_vars = sub_vars - bound_vars
        self.bound_vars = bound_vars

    def to_code_rep(self):
        if (rest := self.rest) is not None:
            rest_code = rest.to_code_rep()
        else:
            rest_code = None
        return ComprehensionContainerCode(self.vars, self.container.to_code_rep(), rest_code)

    def describe(self, parent_binding=None, for_text=' FOR ', in_text=' IN '):
        def type_str(v):
            return f':{t}' if (t := v.type) not in (M_ANY, M_UNKNOWN) else ''

        rest_desc = rest.describe() if (rest := self.rest) is not None else ''
        return f'{for_text}{", ".join(f"{v.name}{type_str(v)}" for v in self.vars)}{in_text}{self.container.describe()}{rest_desc}'

    def arguments(self):
        if (rest := self.rest) is None:
            return self.container,
        return self.container, rest

    def with_argument(self, index, arg):
        if index == 0:
            return ComprehensionContainer(self.vars, arg, self.rest)
        if index == 1 and (rest := self.rest) is not None:
            return ComprehensionContainer(self.vars, self.container, arg)
        raise IndexError('Index for ComprehensionContainer out of bounds')

    def substitute(self, substitutions: Mapping[QualifiedName, 'FormalContent']) -> 'FormalContent':
        args = self.arguments()
        new_args = [arg.substitute(substitutions) if arg is not None else arg
                    for arg in args]
        vars = [new.name if new is not None else old
                for old, new in zip(self.vars, (substitutions.get(v) for v in self.vars))]
        return type(self)(vars, *new_args)

    def substitute_bound_vars(self) -> ('FormalContent', Mapping[QualifiedName, 'FormalContent']):
        # Change bound non-lexically-bound variables by adding scope level to the lexical path
        new_bound_vars = [FormalContent.fresh_name(var, 's') if not var.is_lexically_scoped() else None
                          for var in self.bound_vars]
        if isinstance(self.container, RangeExpr) and len(self.bound_vars) == 1:
            new_bound_vars[0] = new_bound_vars[0].with_type(MRange(self.container.start, self.container.stop))
        changed_vars = {v: MathVariable(n) for v, n in zip(self.bound_vars, new_bound_vars)
                        if n is not None}
        if not changed_vars:
            return self, {}
        result = self.substitute(changed_vars)
        return result, changed_vars


@intern_expr
class ComprehensionCondition(ComprehensionElement):
    has_operator = False

    def __init__(self, condition, rest: Optional[ComprehensionElement] = None):
        super().__init__()
        self.condition = condition
        self.rest = rest
        sub_vars = condition.free_vars
        bound_vars = frozenset()
        if rest is not None:
            sub_vars = rest.free_vars
            bound_vars = bound_vars | rest.bound_vars
        self.free_vars = sub_vars
        self.bound_vars = bound_vars

    def to_code_rep(self):
        if (rest := self.rest) is not None:
            rest_code = rest.to_code_rep()
        else:
            rest_code = None
        return ComprehensionConditionCode(self.condition.to_code_rep(), rest_code)

    def describe(self, parent_binding=None, st_text=' S.T. '):
        rest_desc = rest.describe() if (rest := self.rest) is not None else ''
        return f'{st_text}{self.condition.describe()}{rest_desc}'

    def arguments(self):
        if (rest := self.rest) is None:
            return self.condition,
        return self.condition, rest

    def with_argument(self, index, arg):
        if index == 0:
            return ComprehensionCondition(arg, self.rest)
        if index == 1 and (rest := self.rest) is not None:
            return ComprehensionCondition(self.condition, arg)
        raise IndexError('Index for ComprehensionCondition out of bounds')


@intern_expr
class Aggregate(Term):
    own_binding = 145
    units = {'+': 0, '*': 1}

    @disown('op')
    def __init__(self, op, term: FormalContent, container: ComprehensionContainer = None):
        ftype = FUNCTION_TYPES.get(op)
        super().__init__(ftype.result_type if ftype else M_ANY)
        self.op = op
        self.term = term
        self.container = container
        bound_vars = container.bound_vars if container else frozenset()
        self.free_vars = term.free_vars | (container.free_vars if container else frozenset()) - bound_vars
        self.bound_vars = bound_vars

    def to_code_rep(self):
        return AggregateExpr(self.op, self.term.to_code_rep(),
                             self.container.to_code_rep() if self.container else None)

    def describe(self, parent_binding=None):
        op = f'{AGGREGATE_SYMBOL_MAP[self.op]} '
        cont = self.container.describe(self.own_binding) if self.container else ''
        return self.parenthesize(f'{op}{self.term.describe()}{cont}', parent_binding)

    def operator(self):
        return self.op

    def arguments(self):
        if self.container is None:
            return self.term,
        return self.term, self.container

    def with_argument(self, index, arg):
        if index == 0:
            return Aggregate(self.op, arg, self.container)
        if index == 1:
            return Aggregate(self.op, self.term, arg)
        raise IndexError('Index for Aggregate must be 0 or 1')


# TODO: subclass Condition instead?
@intern_expr
class Quantifier(Term):
    own_binding = 120

    def __new__(cls, kind, formula: Union[Condition, Quantity], container: ComprehensionContainer, unique=False,
                no_added_scope=False):
        if kind == FOR_ALL_SYMBOL and formula.is_eq(TRUE_AS_QUANTITY):
            return TRUE_AS_QUANTITY
        result = super().__new__(cls)
        return result

    @disown('kind', 'unique', 'no_added_scope')
    def __init__(self, kind, formula: Union[Condition, Quantity], container: ComprehensionContainer, unique=False,
                 no_added_scope=False):
        super().__init__(M_BOOLEAN)
        if not no_added_scope:
            # Add lexical scope to bound variables if necessary
            container, substitutions = container.substitute_bound_vars()
            if substitutions:
                formula = formula.substitute(substitutions)
        self.kind = kind
        self.formula = formula
        self.container = container
        self.unique = unique
        bound_vars = container.bound_vars
        self.free_vars = formula.free_vars - bound_vars
        self.bound_vars = bound_vars
        # for interning
        self.no_added_scope = no_added_scope

    def with_formula(self, formula, force=False):
        if not force and self.formula != TRUE_AS_QUANTITY:
            raise Exception('Quantifier already has a formula')
        return Quantifier(self.kind, formula, self.container, unique=self.unique, no_added_scope=True)

    def to_code_rep(self):
        return QuantifierExpr(self.kind, self.formula.to_code_rep(), self.container.to_code_rep(),
                              unique=self.unique)

    def describe(self, parent_binding=None):
        op = f'{self.kind}'
        if self.unique:
            op += '!'
        return self.parenthesize(f'{op}{self.container.describe(for_text="", in_text=f" {ELEMENT_OF_SYMBOL} ")}. '
                                 f'{self.formula.describe(self.own_binding)}',
                                 parent_binding)

    def operator(self):
        return self.kind + ('!' if self.unique else '')

    def arguments(self):
        return self.formula, self.container

    def with_argument(self, index, arg):
        if index == 0:
            return Quantifier(self.kind, arg, self.container, unique=self.unique, no_added_scope=True)
        if index == 1:
            return Quantifier(self.kind, self.formula, arg, unique=self.unique, no_added_scope=True)
        raise IndexError('Index for Quantifier must be 0 or 1')

    def with_arguments(self, args: Sequence[Term]):
        return type(self)(self.kind, args[0], args[1], unique=self.unique, no_added_scope=True)

    def skolemize(self, universals: Sequence[QualifiedName], var_factory: 'MathVariableFactory') -> FormalContent:
        if self.kind == FOR_ALL_SYMBOL:
            extended_universals = (*universals, *(var for var in self.container.vars if var not in universals))
            return Quantifier(self.kind,
                              self.formula.skolemize(extended_universals, var_factory),
                              self.container.skolemize(extended_universals, var_factory),
                              no_added_scope=True)
        # assert self.kind == EXISTS_SYMBOL
        substitutions = {var: var_factory.create_math_variable(
            base_name=var,
            mtype=var.type,
            indexes=[MathVariable(idx) for idx in universals])
            for var in self.container.vars}
        # TODO: test the case of a container that is not RealSpace
        formula = self.formula if isinstance(self.container.container, RealSpace) else m_and(self.container,
                                                                                             self.formula)
        return formula.substitute(substitutions).skolemize([v for v in universals if v not in self.container.vars],
                                                           var_factory)


@intern_expr
class Cast(Term):
    has_operator = False

    @disown('type')
    def __init__(self, type: MType, term: Term):
        super().__init__(type)
        self.term = term

    def to_code_rep(self):
        return CastExpr(self.type, self.term.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(f'CAST({self.type}; {self.term.describe()})', parent_binding)

    def arguments(self):
        return self.term,

    def with_argument(self, index, arg):
        return Cast(self.type, arg)


@intern_expr
class LambdaExpression(Term):
    has_operator = False

    @disown('vars')
    def __init__(self, vars: Sequence[QualifiedName], body: Union[Term, Condition]):
        self.vars = vars or frozenset()
        if isinstance(body, Condition):
            super().__init__(MFunctionType([var.type for var in self.vars], M_BOOLEAN))
        elif isinstance(body, Term):
            super().__init__(MFunctionType([var.type for var in self.vars], body.get_type()))
        self.body = body
        sub_vars = body.free_vars
        bound_vars = frozenset(vars) if vars else frozenset()
        self.free_vars = sub_vars - bound_vars
        self.bound_vars = bound_vars

    def to_code_rep(self):
        return LambdaExpressionExpr(self.vars, self.body.to_code_rep())

    def arguments(self):
        return self.body,

    def describe(self, parent_binding=None):
        args_desc = f"({', '.join(arg.describe() for arg in self.vars)})"
        return self.parenthesize(f'{args_desc} -> {self.body.describe(self.own_binding)}', parent_binding)

    def with_argument(self, index, arg):
        if index == 0:
            return LambdaExpression(self.vars, arg)
        raise IndexError('Index for Lambda must be 0')


def apply_lambda(lfunc: LambdaExpression, *args: Term) -> Term:
    if len(lfunc.vars) != len(args):
        raise Exception(
            f'LambdaExpression has {len(lfunc.vars)} args, but {len(args)} supplied: {lfunc}({", ".join(args)})')
    return lfunc.body.substitute({v: a for v, a in zip(lfunc.vars, args)})


@intern_expr
class FunctionApplication(Term):
    def __new__(cls, function: QualifiedName, args: Sequence[FormalContent], method_target: Optional[Term] = None,
                constructor=False, canonicalize=True):
        if not canonicalize:
            result = super().__new__(cls)
        elif function == PLUS_QN:
            reduced_args = [e for e in args if e != ZERO_AS_QUANTITY]
            if not reduced_args:
                return ZERO_AS_QUANTITY
            if len(reduced_args) == 1:
                result = reduced_args[0]
                result.already_initialized = True
                return result
            # TODO: apply similar simplifications to other arithmetical operators
            quantities = [e for e in args if isinstance(e, Quantity)]
            result = super().__new__(cls)
            if len(quantities) > 1:
                non_quantities = [e for e in args if not isinstance(e, Quantity)]
                qsum = Quantity(sum(q.value for q in quantities))
                if non_quantities:
                    result.__init__(function, tuple(chain(non_quantities, [qsum])))
                    result.already_initialized = True
                else:
                    result = qsum
                    return result
            elif len(args) != len(reduced_args):
                result.__init__(function, reduced_args, method_target)
                result.already_initialized = True
        elif function == TIMES_QN:
            reduced_args = [e for e in args if e != ONE_AS_QUANTITY]
            if not reduced_args:
                return ONE_AS_QUANTITY
            if any(True for e in args if e.is_eq(ZERO_AS_QUANTITY)):
                return ZERO_AS_QUANTITY
            if len(reduced_args) == 1:
                result = reduced_args[0]
                result.already_initialized = True
                return result
            result = super().__new__(cls)
            if len(args) != len(reduced_args):
                result.__init__(function, reduced_args, method_target)
                result.already_initialized = True
        elif function == MINUS_QN:
            reduced_args = [e for e in args if e != ZERO_AS_QUANTITY]
            if not reduced_args:
                return ZERO_AS_QUANTITY
            result = super().__new__(cls)
            if len(args) != len(reduced_args):
                if len(reduced_args) == 1:
                    if args[0] != ZERO_AS_QUANTITY:
                        result = reduced_args[0]
                        result.already_initialized = True
                        return result
                result.__init__(function, reduced_args, method_target)
                result.already_initialized = True
        else:
            result = super().__new__(cls)
        return result

    @disown('function', 'constructor', 'canonicalize')
    def __init__(self, function: QualifiedName, args: Sequence[FormalContent], method_target: Optional[Term] = None,
                 constructor=False, canonicalize=True):
        try:
            if self.already_initialized:
                return
        except AttributeError:
            pass
        assert isinstance(function, QualifiedName)  # DEBUG
        self.function = function
        # TODO: compute type based on argument types
        if constructor:
            result_type = MClassType(function)
        else:
            result_type = function_type(function).result_type
        super().__init__(result_type)
        # FIXME: doesn't support expressions as functions
        self.own_binding = FUNCTION_PRECEDENCE_MAP.get(function.name)
        for i, arg in enumerate(args):
            if isinstance(arg, LogicalCombination):
                args[i:i + 1] = arg.elements
                i += len(arg.elements)
        self.args = args
        self.method_target = method_target
        self.constructor = constructor
        free_vars = frozenset(chain.from_iterable(a.free_vars for a in args))
        if method_target:
            free_vars = free_vars | method_target.free_vars
        self.free_vars = free_vars

    def to_code_rep(self):
        target = mt.to_code_rep() if (mt := self.method_target) is not None else None
        return FunctionApplExpr(self.function, [a.to_code_rep() for a in self.args],
                                method_target=target,
                                constructor=self.constructor)

    def describe(self, parent_binding=None):
        target = f'{mt.describe()}.' if (mt := self.method_target) is not None else ''
        constructor_call = 'new ' if self.constructor else ''
        function_name = self.function.name
        if mt is None and is_math_name(self.function) and len(function_name) == 1 and len(self.args) > 1:
            arg_bindings = (repeat(self.own_binding) if function_name in ASSOCIATIVE_OPERATORS
                            else chain([self.own_binding], repeat(self.own_binding + 1)))
            result = f' {function_name} '.join(arg.describe(parent_binding=binding) for
                                               arg, binding in zip(self.args, arg_bindings))
        else:
            args_desc = ', '.join(arg.describe() for arg in self.args)
            result = f'{constructor_call}{target}{function_name}({args_desc})'
        return self.parenthesize(result, parent_binding)

    def without_target(self):
        return FunctionApplication(self.function, self.args, None)

    def operator(self):
        return self.function

    def arguments(self) -> Sequence[FormalContent]:
        if self.method_target is not None:
            return [self.method_target, *self.args]
        else:
            return self.args

    def with_function(self, function: QualifiedName):
        return FunctionApplication(function, self.args, self.method_target)

    def with_argument(self, index, arg):
        if self.method_target is not None:
            if index == 0:
                return FunctionApplication(self.function, self.args, method_target=arg)
            return FunctionApplication(self.function, with_element(self.args, index - 1, arg),
                                       method_target=self.method_target)
        return FunctionApplication(self.function, with_element(self.args, index, arg))

    def with_arguments(self, args: Sequence[Term]):
        if self.method_target is not None:
            return FunctionApplication(self.function, args[1:], method_target=args[0])
        return FunctionApplication(self.function, args)


# Shorthand for creating a FunctionApplication from a QualifiedName
def qn_call(self: QualifiedName, *args, **kwargs):
    return FunctionApplication(self, args, **kwargs)


QualifiedName.__call__ = qn_call


@intern_expr
class Percentage(Term):
    own_binding = 190
    # Treating number as operator
    has_operator = True

    @disown('pct')
    def __init__(self, pct, whole: Term):
        super().__init__(M_NUMBER)
        self.pct = pct
        self.whole = whole
        self.free_vars = whole.free_vars

    def describe(self, parent_binding=None):
        return self.parenthesize(f'{self.pct}% OF {self.whole.describe(self.own_binding)}', parent_binding)

    def operator(self):
        return self.pct

    def arguments(self):
        return self.whole

    def with_argument(self, index, arg):
        if index == 0:
            return Percentage(self.pct, arg)
        raise IndexError('Index for Percentage must be 0')


@intern_expr
class Between(Term):
    has_operator = False

    def __init__(self, value: Term, lower_bound: Term, upper_bound: Term):
        super().__init__(M_BOOLEAN)
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.free_vars = value.free_vars | lower_bound.free_vars | upper_bound.free_vars

    def to_code_rep(self):
        return BetweenExpr(self.lower_bound.to_code_rep(), self.value.to_code_rep(), self.upper_bound.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(f'{self.lower_bound.describe(Comparison.own_binding)} {LE_SYMBOL} '
                                 f'{self.value.describe(Comparison.own_binding)} '
                                 f'{LE_SYMBOL} {self.upper_bound.describe(Comparison.own_binding)}',
                                 parent_binding)

    def arguments(self):
        return self.value, self.lower_bound, self.upper_bound

    def with_argument(self, index, arg):
        return Between(*with_element(self.arguments(), index, arg))


@intern_expr
class GeneralSet(Term):
    has_operator = False

    def __init__(self, elements):
        super().__init__(MSetType(MUnionType({e.get_type() for e in elements})))
        self.elements = elements
        self.free_vars = frozenset(chain.from_iterable(e.free_vars for e in self.elements))

    def to_code_rep(self):
        return SetExpr([e.to_code_rep() for e in self.elements])

    def describe(self, parent_binding=None):
        return self.parenthesize('{' + ', '.join(e.describe(self.own_binding) for e in self.elements) + '}',
                                 parent_binding)

    def arguments(self):
        return self.elements

    def with_argument(self, index, arg):
        return GeneralSet(with_element(self.elements, index, arg))

    def with_arguments(self, args: Sequence[Term]):
        return GeneralSet(args)


@intern_expr
class GeneralSequence(Term):
    has_operator = False

    def __init__(self, elements):
        super().__init__(MCollectionType(MUnionType({e.get_type() for e in elements})))
        self.elements = elements
        self.free_vars = frozenset(chain.from_iterable(e.free_vars for e in self.elements))

    def to_code_rep(self):
        return SequenceExpr([e.to_code_rep() for e in self.elements])

    def describe(self, parent_binding=None):
        return self.parenthesize('[' + ', '.join(e.describe(self.own_binding) for e in self.elements) + ']',
                                 parent_binding)

    def arguments(self):
        return self.elements

    def with_argument(self, index, arg):
        return GeneralSequence(with_element(self.elements, index, arg))

    def with_arguments(self, args: Sequence[Term]):
        return GeneralSequence(args)


@intern_expr
class Stream(Term):
    """
    A potentially lazy linear collection
    """
    has_operator = False

    @disown('element_type')
    def __init__(self, term: FormalContent, container: ComprehensionContainer, element_type=M_ANY):
        """
        :param term: expression giving values of the stream
        :param container: the container over which the stream iterates
        :param element_type: element type of the result stream
        """
        super().__init__(MStreamType(element_type))
        self.term = term
        self.container = container
        bound_vars = container.bound_vars
        self.free_vars = term.free_vars | container.free_vars - bound_vars
        self.bound_vars = bound_vars

    def to_code_rep(self):
        return StreamExpr(self.term.to_code_rep(), self.container.to_code_rep())

    def describe(self, parent_binding=None):
        return self.parenthesize(f'STREAM({self.term.describe()}{self.container.describe()})', parent_binding)

    def arguments(self):
        return self.term, self.container

    def with_argument(self, index, arg):
        if index == 0:
            return Stream(arg, self.container, element_type=self.type)
        if index == 1:
            return Stream(self.term, arg, element_type=self.type)
        raise IndexError('Index for Stream must be 0 or 1')


INTERNAL_DEFS_TYPE = Sequence[Union['FunctionDefinitionExpr', 'ClassDefinitionExpr', 'MathTypeDeclaration']]


class FunctionDefinitionExpr(FormalContent):
    """
    A function or method definition, similar to a lambda expression.  The body of the function is an expression,
    so only functional style (no side effects) is possible.

    If a type isn't available for a parameter or return, use M_ANY.
    """

    @disown('name', 'typed_parameters', 'return_type', 'lexical_path', 'func_doc_string', 'method_target',
            'is_static_method', 'is_class_method')
    def __init__(self, name: QualifiedName, typed_parameters: Sequence[ParameterDescriptor], return_type: MType,
                 func_doc_string: Optional[str], body: Term, method_target: Optional[ParameterDescriptor],
                 lexical_path, defs: INTERNAL_DEFS_TYPE = (), decorators: Sequence[Term] = (),
                 is_static_method=False, is_class_method=False):
        super().__init__()
        self.name = name
        self.func_doc_string = func_doc_string
        self.typed_parameters = typed_parameters
        self.return_type = return_type
        self.body = body
        self.method_target = method_target
        self.defs = defs
        all_params = typed_parameters
        if method_target:
            all_params = [method_target] + all_params
        parameter_types = [param.type for param in all_params]
        self.type = MFunctionType(parameter_types, return_type)
        self.decorators = decorators
        self.lexical_path = lexical_path
        self.is_static_method = is_static_method
        self.is_class_method = is_class_method

    def get_defs(self):
        if isinstance(self.defs, BodyExpr):
            return self.defs.defs
        return self.defs

    def get_value(self):
        if isinstance(self.body, BodyExpr):
            return self.body.value
        return self.body

    def describe(self, parent_binding=None):
        this = f'{target.name}.' if (target := self.method_target) is not None else ''
        params = ', '.join(f'{param.name}:{param.type}' for param in self.typed_parameters)
        internals = '; '.join(d.describe() for d in self.defs)
        int_desc = f'[{internals}] ' if internals else ''
        dec = (''.join(f'@{d.describe()} ' for d in decorators)) if (decorators := self.decorators) else ''
        return (f'DEFINE {dec}{this}{self.name.name}({params}): {self.func_doc_string or ""}{self.return_type} '
                f'{int_desc}= {self.body.describe()}')

    def to_code_rep(self):
        if isinstance(self.body, BodyExpr):
            value = self.body.value.to_code_rep()
            doc = self.body.element_doc_string
            defs = [d.to_code_rep() for d in self.body.defs]
        else:
            value = self.body.to_code_rep()
            doc = None
            defs = ()
        return AbstractFunctionDefinition(self.name, self.typed_parameters, self.return_type, value,
                                          func_doc_string=doc, method_target=self.method_target, defs=defs,
                                          decorators=[d.to_code_rep() for d in self.decorators])

    def arguments(self):
        return [self.body, *self.defs]

    def with_argument(self, index, arg):
        # TODO: implement this
        raise NotImplementedError('Rule substitution not implemented for FunctionDefinitionExpr')


class ClassDefinitionExpr(FormalContent):
    """
    A class definition, which contains methods and fields.
    """
    type = M_TYPE

    @disown('name', 'is_dataclass')
    def __init__(self, name: QualifiedName, superclasses: Sequence[Term],
                 defs: Optional[Union[list, tuple, 'BodyExpr']] = None,
                 is_dataclass=False, decorators=(), fields: Sequence['MathTypeDeclaration'] = ()):
        super().__init__()
        self.name = name
        self.superclasses = superclasses
        self.defs = defs
        self.is_dataclass = is_dataclass
        self.fields = fields
        self.decorators = decorators

    def as_dataclass(self, fields, removed_defs=None):
        if removed_defs is None:
            removed_defs = fields
        defs = (self.defs.without(removed_defs) if isinstance(self.defs, BodyExpr)
                else [d for d in self.defs if d not in fields])
        return ClassDefinitionExpr(self.name, self.superclasses,
                                   defs,
                                   is_dataclass=True,
                                   fields=fields,
                                   decorators=self.decorators)

    def get_defs(self):
        if isinstance(self.defs, BodyExpr):
            return self.defs.defs
        return self.defs

    def describe(self, parent_binding=None):
        supers = '(' + ', '.join(str(s) for s in sup) + ')' if (sup := self.superclasses) else ''
        if isinstance(self.defs, BodyExpr):
            if self.defs.value is not None:
                raise Exception("Class can't have return statement")
            doc = f'DOC="{doc_string}"' if (doc_string := self.defs.element_doc_string) else ''
            defs = self.defs.defs
        else:
            doc = ''
            defs = self.defs or []
        internals = '; '.join(d.describe() for d in list(self.fields if self.fields else []) + defs)
        int_desc = f' {internals}' if internals else ''
        dec = (''.join(f'@{d.describe()} ' for d in decorators)) if (decorators := self.decorators) else ''
        return f'{"DATA-" if self.is_dataclass else ""}CLASS {dec}{self.name.name}{supers}: {doc}{int_desc}'

    def to_code_rep(self):
        if isinstance(self.defs, BodyExpr):
            if self.defs.value is not None:
                raise Exception("Class can't have return statement")
            doc = self.defs.element_doc_string
            defs = [d.to_code_rep() for d in self.defs.defs]
        else:
            doc = None
            defs = [d.to_code_rep() for d in self.defs]
        if fields := self.fields:
            defs = [f.to_code_rep() for f in fields] + defs
        # TODO: forward is_dataclass as well
        return AbstractClassDefinition(self.name, [sc.to_code_rep() for sc in self.superclasses], doc, defs,
                                       decorators=[d.to_code_rep() for d in self.decorators])

    def arguments(self):
        return body.defs if isinstance(body := self.defs, BodyExpr) else body

    def with_argument(self, index, arg):
        raise NotImplementedError('Rule substitution not implemented for ClassDefinitionExpr')


class BodyExpr(FormalContent):
    """
    A function or class body, with (optional) doc string, internal definitions, and (for functions/methods only) value.
    """

    @disown('element_doc_string')
    def __init__(self, element_doc_string: str = None, defs: INTERNAL_DEFS_TYPE = (), value=None):
        super().__init__()
        self.element_doc_string = element_doc_string
        self.defs = defs
        self.value = value

    def without(self, fields: Sequence['MathTypeDeclaration']):
        return BodyExpr(element_doc_string=self.element_doc_string,
                        defs=[d for d in self.defs if d not in fields],
                        value=self.value)

    def describe(self, parent_binding=None):
        doc_str = f' DOC="{doc}"' if (doc := self.element_doc_string) is not None else ''
        def_str = f' DEFS=[{"; ".join(d.describe() for d in defs)}]' if (defs := self.defs) else ''
        ret_str = f' VALUE="{value.describe()}"' if (value := self.value) is not None else ''
        return f'BODY{doc_str}{def_str}{ret_str}'

    def to_code_rep(self):
        raise Exception('BodyExpr should not be translated directly to code')

    def arguments(self):
        if self.value:
            return tuple(self.defs) + (self.value,)
        return self.defs

    def with_argument(self, index, arg):
        raise NotImplementedError('Rule substitution not implemented for BodyExpr')


@intern_expr
class LetExpr(Term):
    # FIXME!! change var to QN
    @disown('var', 'type_name')
    def __init__(self, var: str, value: Term, body: Term, type_name: MType = M_ANY):
        super().__init__(type_name)
        self.var = var
        self.value = value
        self.body = body
        self.free_vars = value.free_vars | body.free_vars
        self.bound_vars = frozenset([var])

    def describe(self, parent_binding=None):
        return f'LET {self.var} = {self.value.describe()} IN {self.body.describe()}'

    # FIXME: override to_code_rep()

    def operator(self):
        return self.var

    def arguments(self):
        return self.value, self.body

    def with_argument(self, index, arg):
        return LetExpr(*with_element(self.arguments(), index, arg))


@intern_expr
class MathTypeDeclaration(FormalContent):
    has_operator = True

    # FIXME!! change var to QN
    @disown('var', 'type', 'primary_key', 'is_domain', 'is_solution_var')
    def __init__(self, var: str, type: MType, primary_key=False, is_domain=False, is_solution_var=False):
        super().__init__()
        self.var = var
        self.type = type
        self.primary_key = primary_key
        self.is_domain = is_domain
        self.is_solution_var = is_solution_var

    def describe(self, parent_binding=None):
        pk = ' (Primary key)' if self.primary_key else ''
        return f'{self.var}: {self.type}{pk}'

    def to_code_rep(self):
        # Note that primary key information is removed (not really necessary to run the program)
        return AbstractTypeDeclaration(self.var, self.type)

    def operator(self):
        return self.var

    def arguments(self):
        return ()
        # raise NotImplementedError('Pattern matching not implemented for MathTypeDeclaration')

    def with_argument(self, index, arg):
        raise NotImplementedError('Rule substitution not implemented for MathTypeDeclaration')

    def with_options(self, primary_key=False, is_domain=False, is_solution_var=False):
        return MathTypeDeclaration(self.var, self.type,
                                   primary_key=primary_key, is_domain=is_domain, is_solution_var=is_solution_var)


@intern_expr
class InitializedVariable(FormalContent):
    has_operator = True

    def __init__(self, var: Union[MathVariable, MathTypeDeclaration], init: Term):
        super().__init__()
        self.var = var
        self.init = init
        self.free_vars = init.free_vars

    def describe(self, parent_binding=None):
        return f'{self.var.describe()} = {self.init.describe()}'

    def to_code_rep(self):
        if isinstance(var := self.var, MathTypeDeclaration):
            var_name = var.var
            vtype = var.type
        else:
            # MathVariable
            var_name = var.name
            vtype = M_ANY
        return AbstractTypeDeclaration(var_name, vtype, init=self.init.to_code_rep())

    def operator(self):
        return self.var

    def arguments(self):
        return [self.init]

    def with_argument(self, index, arg):
        if index != 0:
            raise IndexError('Index for Negate must be 0')
        return InitializedVariable(self.var, arg)


@intern_expr
class NamedArgument(Term):
    has_operator = False

    @disown('name')
    def __init__(self, name, expr: Term):
        super().__init__(expr.type)
        self.name = name
        self.expr = expr

    def describe(self, parent_binding=None):
        return f'{self.name}={self.expr.describe()}'

    def to_code_rep(self):
        return AbstractNamedArg(self.name, self.expr.to_code_rep())

    def with_argument(self, index, arg):
        raise NotImplementedError('Pattern matching not implemented for NamedArgument')


class MathModule(FormalContent):
    def __init__(self, contents: Sequence[FormalContent]):
        super().__init__()
        self.contents = contents
        function_pattern = ClassPattern(FunctionDefinitionExpr)
        self.all_functions = {(func := res[0]).name: func for res in find_recursive_matches(function_pattern, self)}
        class_pattern = ClassPattern(ClassDefinitionExpr)
        self.classes = {(cls := res[0]).name: cls for res in find_recursive_matches(class_pattern, self)}
        call_pattern = ClassPattern(FunctionApplication)
        self.method_calls = list(call[0] for call in find_recursive_matches(call_pattern, self)
                                 if call[0].method_target is not None)
        # TODO: make the linearize function a parameter, so that languages with other MROs can be supported in future
        self.all_superclasses = linearize({clsname: [sc.name for sc in cls.superclasses]
                                           for clsname, cls in self.classes.items()})
        self.class_members: MutableMapping[QualifiedName, Mapping[str, QualifiedName]] = {}

    def add_class_members(self, class_members: Mapping[QualifiedName, Mapping[str, QualifiedName]]):
        self.class_members.update(class_members)

    def add_module(self, module: 'MathModule'):
        """
        Add information from a module used by this one.

        N.B. Should only be called on main module (containing the optimization problem).
        """
        self.all_functions.update(module.all_functions)
        self.classes.update(module.classes)
        self.method_calls.extend(module.method_calls)
        self.all_superclasses = linearize({clsname: [sc.name for sc in cls.superclasses]
                                           for clsname, cls in self.classes.items()})
        self.add_class_members(module.class_members)

    def member_of_class(self, class_name: QualifiedName, member_name: str) -> QualifiedName:
        superclasses = self.all_superclasses[class_name]
        return next((qn for sc in superclasses
                     if (qn := self.class_members[sc].get(member_name)) is not None),
                    None)

    def all_class_methods(self, class_name: QualifiedName) -> Sequence[FunctionDefinitionExpr]:
        superclasses = self.all_superclasses[class_name]
        members = sorted({m for sc in superclasses for m in self.class_members[sc]})
        return [method for m in members
                if (method := self.all_functions.get(self.member_of_class(class_name, m))) is not None]

    def describe(self, parent_binding=None):
        contents = '; '.join(c.describe() for c in self.contents if c is not None)
        return f'Math-Module({contents})'

    def to_code_rep(self):
        return CompilationUnit([c.to_code_rep() for c in self.contents])

    def arguments(self):
        return self.contents

    def with_argument(self, index, arg):
        return MathModule(with_element(self.contents, index, arg))


# TODO: add rules to eliminate expressions like $y - ε + ε?  Would require complex arithmetic reasoning...
@intern_expr
class Epsilon(Term):
    """
    An infinitesimal value
    """

    @disown('approximation')
    def __init__(self, approximation=None):
        super().__init__(M_NUMBER)
        self.approximation = approximation

    def describe(self, parent_binding=None):
        return EPSILON_SYMBOL + (approx if (approx := self.approximation) is not None else '')

    def to_code_rep(self):
        if (approx := self.approximation) is not None:
            return NumberExpr(approx)
        # FIXME: translate as infinitesimal
        return NamedConstant('epsilon', M_NUMBER)

    def arguments(self):
        return [approx] if (approx := self.approximation) is not None else []

    def with_argument(self, index, arg):
        if arg != 0:
            raise IndexError('Index for Epsilon must be 0')
        return Epsilon(arg)


@intern_expr
class TypeAlias(FormalContent):
    @disown('alias', 'mtype')
    def __init__(self, alias: str, mtype: MType):
        super().__init__()
        self.alias = alias
        self.mtype = mtype

    def describe(self, parent_binding=None):
        return f'ALIAS {self.alias} AS {self.mtype}'

    def with_argument(self, index, arg):
        return self


class DummyTerm(Term):
    """
    A dummy term, must not be used to generate code!
    """

    def __init__(self, role: Tuple[str, ...], serial=None):
        super().__init__(M_BOTTOM)
        self.role = role
        self.serial = serial

    def describe(self, parent_binding=None):
        serial = f'#{self.serial}' if self.serial is not None else ''
        return f'<DUMMY{serial}: {" ".join(self.role)}>'

    def to_code_rep(self):
        raise Exception(f'Dummy term must not be used to generate code')

    def with_argument(self, index, arg):
        return self


@intern_expr
class TypeExpr(FormalContent):
    """
    A type encapsulated as a FormalContent.
    """

    @disown('mtype')
    def __init__(self, mtype: MType):
        super().__init__()
        self.mtype = mtype

    def describe(self, parent_binding=None):
        return f'(Type: {self.mtype})'

    def with_argument(self, index, arg: Term):
        raise Exception(f'TypeExpr must not be used to generate code')


BOOLEAN_TYPE_EXPR = TypeExpr(M_BOOLEAN)


@intern_expr
class RangeExpr(Term):
    # TODO: replace by TypeExpr of range type when created
    @disown('start', 'stop')
    def __init__(self, start: int, stop: int):
        super().__init__(MCollectionType(MRange(start, stop)))
        self.start = start
        self.stop = stop

    def describe(self, parent_binding=None):
        return f'Range({self.start}, {self.stop})'

    def with_argument(self, index, arg):
        raise Exception('RangeExpr.with_arguments() not supported')

    def to_code_rep(self):
        return RangeCode(self.start, self.stop)

    def as_range(self):
        return range(self.start, self.stop)

    def __iter__(self):
        return iter(self.as_range())


@intern_expr
class RealSpace(Term):
    @disown('n')
    def __init__(self, n: int = 1):
        super().__init__(MCollectionType(M_NUMBER))
        self.n = n

    def describe(self, parent_binding=None):
        return f'R^{self.n}'

    def with_argument(self, index, arg):
        return self


@intern_expr
class DomainDim(Term, DeferredType):
    """
    A term representing a domain that is as-yet unknown.
    """

    def __new__(cls, domain_of: Term):
        if isinstance(domain_of.type, MRange):
            return RangeExpr(domain_of.type.start, domain_of.type.stop)
        return super().__new__(cls)

    def __init__(self, domain_of: Term):
        super().__init__(M_ANY)
        self.domain_of = domain_of

    def describe(self, parent_binding=None):
        return f'(DomainDim from {self.domain_of.describe()})'

    def arguments(self):
        return self.domain_of,

    def with_argument(self, index, arg):
        if index != 0:
            raise IndexError('Index for DomainDim must be 0')
        return DomainDim(arg)


@intern_expr
class DefinedBy(Condition):
    """
    Dummy objects that reduce to true but define a dependency between a set of inputs and a set of outputs
    """

    def __init__(self, inputs: Sequence[Term], outputs: Sequence[Term]):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def describe(self, parent_binding=None):
        return (f'DefinedBy(inputs={", ".join(v.describe() for v in self.inputs)}, '
                f'outputs={", ".join(v.describe() for v in self.inputs)})')

    def arguments(self):
        return *self.inputs, *self.outputs

    def with_argument(self, index, arg):
        if index < len(self.inputs):
            return type(self)([arg if i == index else inp for i, inp in enumerate(self.inputs)], self.outputs)
        else:
            out_index = index - len(self.inputs)
            return type(self)(self.inputs, [arg if i == out_index else outp for i, outp in enumerate(self.outputs)])

    def to_code_rep(self):
        return BooleanExpr(True)


@visitor_for(FormalContent, collect_results=False)
class ExprAbstractNonCollectingVisitor:
    pass


class MathVariableFactory(ABC):
    @abstractmethod
    def create_math_variable(self, base_name: QualifiedName, mtype: MType, indexes: Optional[Sequence[Term]] = None
                             ) -> Union[MathVariable, IndexedMathVariable]:
        """
        Return a new MathVariable, whose name is created from the base_name.  The role,
        if given, must be unique.  Either base_name or role must be provided.

        :param base_name: suggested name
        :param mtype: type of new variable
        :param indexes: optional indexes for the result, which will be an IndexedMathVariable
        """


def coerce_to_quantity(x):
    if isinstance(x, FormalContent):
        return x
    if isinstance(x, (bool, Number, str)):
        return Quantity(x)
    raise Exception(f'Cannot convert {x} to a quantity')


def m_min(*exprs: Term) -> Term:
    return FunctionApplication(MIN_QN, [coerce_to_quantity(e) for e in exprs])


def m_max(*exprs: Term) -> Term:
    return FunctionApplication(MAX_QN, [coerce_to_quantity(e) for e in exprs])


def m_and(*exprs: Condition) -> Condition:
    return LogicalOperator(AND_QN, exprs)


def m_implies(antecedent: Condition, consequent: Condition):
    return LogicalOperator(IMPLIES_QN, [antecedent, consequent])


def m_ifte(cond: Condition, pos: Term, neg: Term):
    return IFTE(cond, coerce_to_quantity(pos), coerce_to_quantity(neg))


def m_for_all(vars: Sequence[MathVariable], container: Term, formula: Condition):
    # TODO: do the same for other factory methods
    if isinstance(vars, FormalContent):
        vars = [vars]
    return Quantifier(FOR_ALL_SYMBOL, formula, ComprehensionContainer([v.name for v in vars], container))


def m_exists(vars: Sequence[MathVariable], container: Term, formula: Condition):
    return Quantifier(EXISTS_SYMBOL, formula, ComprehensionContainer([v.name for v in vars], container))


def m_sum(var: MathVariable, container: Term, term: Term):
    return Aggregate('+', term, ComprehensionContainer([var.name], container))
