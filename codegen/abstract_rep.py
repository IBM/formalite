from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from numbers import Number
from operator import attrgetter
from typing import Union, List, Sequence, Optional, AbstractSet

from itertools import groupby, chain

from codegen.parameters import ParameterDescriptor
from codegen.utils import intern, disown
from math_rep.constants import ELEMENT_OF_SYMBOL, NOT_SYMBOL, AND_SYMBOL, OR_SYMBOL, IMPLIES_SYMBOL, LE_SYMBOL, \
    GE_SYMBOL, NOT_EQUALS_SYMBOL, NOT_ELEMENT_OF_SYMBOL
from math_rep.expression_types import M_BOOLEAN, M_NUMBER, M_STRING, MType, M_UNKNOWN, M_ANY, QualifiedName, \
    as_math_name, is_math_name

ABSTRACT_DEFS_TYPE = Sequence[Union['AbstractFunctionDefinition', 'AbstractClassDefinition', 'AbstractTypeDeclaration']]


# FIXME!!!!! copying FormalContent.free_vars to CodeElement.prog_free_vars means that prog_free_vars must be added to any
# other generation of a CodeElement!

def symmetric_dict(d):
    result = {}
    result.update(d)
    result.update({v: k for k, v in d.items()})
    return result


NOT_OPERATOR = as_math_name(NOT_SYMBOL)
AND_OPERATOR = as_math_name(AND_SYMBOL)
OR_OPERATOR = as_math_name(OR_SYMBOL)
IMPLIES_OPERATOR = as_math_name(IMPLIES_SYMBOL)
PLUS_OPERATOR = as_math_name('+')
MINUS_OPERATOR = as_math_name('-')
TIMES_OPERATOR = as_math_name('*')
LE_OPERATOR = as_math_name(LE_SYMBOL)
GE_OPERATOR = as_math_name(GE_SYMBOL)
NE_OPERATOR = as_math_name(NOT_EQUALS_SYMBOL)
NOT_ELEMENT_OF_OPERATOR = as_math_name(NOT_ELEMENT_OF_SYMBOL)
COMPARISON_NEGATIONS = symmetric_dict({'=': NE_OPERATOR, '<': GE_OPERATOR, '>': LE_OPERATOR,
                                       ELEMENT_OF_SYMBOL: NOT_ELEMENT_OF_OPERATOR})


class Import:
    def __init__(self, name, module, static: bool = False):
        self.name = name
        self.module = module
        self.static = static

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Import) and self.name == o.name and self.module == o.module and self.static == o.static

    def __hash__(self) -> int:
        return self.name.__hash__() + 5 * self.module.__hash__() + 7 * self.static

    def __str__(self):
        static_str = ' (static)' if self.static else ''
        return f'Import("{self.name}", "{self.module}{static_str}")'

    def __repr__(self):
        static_str = ' (static)' if self.static else ''
        return f'Import("{self.name}", "{self.module}{static_str}")'


class CodeFragment:
    def __init__(self, value, body=None, precedence: Optional[int] = None, doc_string: Optional[str] = None,
                 free_vars: AbstractSet[QualifiedName] = frozenset()):
        self.value = value
        self.body = body
        self.precedence = precedence
        self.doc_string = doc_string
        self.free_vars = free_vars


def parenthesize(parent_precedence, child_code: CodeFragment):
    if (child_code.precedence is not None and parent_precedence is not None
            and parent_precedence > child_code.precedence):
        return f'({child_code.value})'
    return child_code.value


class FunctionDefinition:
    pass


class CodeVisitor(ABC):
    def __init__(self, base_name, attribute_mappings=()):
        self.base_name = base_name
        self.imports = set()
        self.dummy_vars = {}
        self.attribute_mappings = attribute_mappings
        self.helpers = []
        self.var_counter = 0
        self.func_counter = 0

    def add_import(self, imp: Import):
        self.imports.add(imp)

    def pretty_imports(self):
        result = []
        for module, contents in groupby(sorted(self.imports, key=attrgetter('module', 'name')), attrgetter('module')):
            result.append(f'{module}: {", ".join(map(attrgetter("name"), contents))}')
        return '\n'.join(result)

    @abstractmethod
    def pretty_function_def(self, helper: FunctionDefinition) -> str:
        pass

    def helper_functions(self) -> List[FunctionDefinition]:
        return self.helpers

    def add_helper_function(self, func: FunctionDefinition):
        self.helpers.append(func)

    def pretty_helpers(self, checkpoint=0):
        # TODO: use string appropriate for specific language
        return '\n\n'.join(self.pretty_function_def(h) for h in self.helpers[checkpoint:])

    def helper_checkpoint(self):
        return len(self.helpers)

    def clear_helpers(self, checkpoint):
        self.helpers[checkpoint:] = []

    def fresh_var(self):
        self.var_counter += 1
        return f'v{self.var_counter}'

    def fresh_func(self):
        self.func_counter += 1
        return f'{self.base_name}_f{self.func_counter}'

    def push_dummy_var(self, dummy_var, expression):
        if dummy_var in self.dummy_vars:
            raise Exception('Double definition of dummy variable')
        self.dummy_vars[dummy_var] = expression

    def pop_dummy_var(self, dummy_var):
        del self.dummy_vars[dummy_var]

    def visit(self, element):
        return element.accept(self)

    @abstractmethod
    def visit_attribute_access(self, attr):
        pass

    @abstractmethod
    def visit_variable_access(self, var):
        pass

    @abstractmethod
    def visit_comparison_expr(self, comp):
        pass

    @abstractmethod
    def visit_number(self, num):
        pass

    @abstractmethod
    def visit_string(self, s):
        pass

    @abstractmethod
    def visit_set_expr(self, s):
        pass

    @abstractmethod
    def visit_set_membership(self, expr):
        pass

    @abstractmethod
    def visit_quantifier(self, exists):
        pass

    @abstractmethod
    def visit_negation(self, expr):
        pass

    @abstractmethod
    def visit_loop_in(self, loop):
        pass

    @abstractmethod
    def visit_concatenation(self, conc):
        pass

    # @abstractmethod
    # def visit_constant(self, constant):
    #     pass

    @abstractmethod
    def visit_aggregate_expr(self, aggregate):
        pass

    @abstractmethod
    def visit_between(self, between):
        pass

    @abstractmethod
    def visit_function_appl_expr(self, appl):
        pass

    @abstractmethod
    def visit_period(self, period):
        pass

    @abstractmethod
    def visit_time(self, time):
        pass

    @abstractmethod
    def visit_dummy_var(self, dummy_var):
        pass

    @abstractmethod
    def visit_logical_expr(self, expr):
        pass

    @abstractmethod
    def visit_predicate_appl(self, appl):
        pass

    @abstractmethod
    def visit_identifier(self, identifier):
        pass

    @abstractmethod
    def visit_temporal_expr(self, temporal):
        pass

    @abstractmethod
    def visit_boolean(self, value):
        pass

    @abstractmethod
    def visit_cells(self, cells):
        pass

    @abstractmethod
    def visit_comprehension_container_code(self, compr):
        pass

    @abstractmethod
    def visit_comprehension_condition_code(self, compr):
        pass

    @abstractmethod
    def visit_stream_expr(self, stream: 'StreamExpr'):
        pass

    @abstractmethod
    def visit_abstract_function_definition(self, func_def: 'AbstractFunctionDefinition'):
        pass

    @abstractmethod
    def visit_abstract_class_definition(self, class_def: 'AbstractClassDefinition'):
        pass

    @abstractmethod
    def visit_abstrat_type_declaration(self, self1):
        pass

    @abstractmethod
    def visit_named_arg(self, arg):
        pass

    @abstractmethod
    def visit_compilation_unit(self, cu):
        pass

    @abstractmethod
    def visit_subscripted_expr(self, sub):
        pass

    @abstractmethod
    def visit_conditional_expr(self, cond):
        pass

    @abstractmethod
    def visit_sequence_expr(self, seq):
        pass

    @abstractmethod
    def visit_assignment(self, assign):
        pass

    @abstractmethod
    def visit_statements(self, seq):
        pass

    @abstractmethod
    def visit_return_statement(self, ret):
        pass

    @abstractmethod
    def visit_data_constant(self, const):
        pass

    @abstractmethod
    def visit_cast(self, cast):
        pass

    @abstractmethod
    def visit_lambda_expression_expr(self, lfunc):
        pass

    @abstractmethod
    def visit_named_constant(self, const):
        pass


class CodeElement(ABC):
    def __init__(self, comment: Optional[str] = None):
        super().__init__()
        self.doc_string = None
        self.comment = comment

    @abstractmethod
    def accept(self, visitor: CodeVisitor):
        pass


class Expression(CodeElement, ABC):
    def negate(self):
        """
        Return an expression with the negated meaning.  Only use the negation symbol if there is no other way.
        """
        result = FunctionApplExpr(NOT_OPERATOR, [self])
        result.doc_string = f'Negation of {self.doc_string}'
        result.type = M_BOOLEAN
        return result


class Identifier(Expression):
    def __init__(self, name: QualifiedName):
        super().__init__()
        assert isinstance(name, QualifiedName)
        self.name = name

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_identifier(self)


class ConstantExpr(Expression):
    """
    Abstract class for all constant expressions
    """


class NumberExpr(ConstantExpr):
    def __init__(self, value: Number):
        super().__init__()
        self.value = value
        self.type = M_NUMBER

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_number(self)


@intern
class NamedConstant(Expression):
    def __init__(self, name: str, ctype: MType):
        super().__init__()
        self.name = name
        self.type = ctype

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_named_constant(self)


class BooleanExpr(ConstantExpr):
    def __init__(self, value: bool):
        super().__init__()
        self.value = value
        self.type = M_BOOLEAN

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_boolean(self)


class DataConstant(Expression):
    # FIXME: unify this with other types of constants above
    def __init__(self, value, type=M_ANY):
        super().__init__()
        self.value = value
        self.type = type

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_data_constant(self)


class StringExpr(ConstantExpr):
    def __init__(self, s: str):
        super().__init__()
        self.value = s
        self.type = M_STRING

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_string(self)


class CastExpr(Expression):
    def __init__(self, cast_type: MType, term: Expression):
        super().__init__()
        self.cast = cast_type
        self.term = term

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_cast_expr(self)


class PeriodExpr(Expression):
    def __init__(self, start: Expression, end: Expression):
        super().__init__()
        self.start = start
        self.end = end

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_period(self)


class TimeExpr(Expression):
    def __init__(self, time: str):
        super().__init__()
        self.time = time

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_time(self)


class SetExpr(Expression):
    def __init__(self, s: Sequence[Expression]):
        super().__init__()
        self.set = s

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_set_expr(self)


class SequenceExpr(Expression):
    def __init__(self, elements: Sequence[Expression]):
        super().__init__()
        self.elements = elements

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_sequence_expr(self)


class VariableAccess(Expression):
    def __init__(self, name: QualifiedName):
        super().__init__()
        self.name = name

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_variable_access(self)


class DummyVariable(Expression):
    """
    A ploceholder for a template variable.

    An object of this class should never be translated to code directly.
    """
    doc_string = ''

    def __init__(self):
        super().__init__()
        self.type = M_UNKNOWN

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_dummy_var(self)


class AttributeAccess(Expression):
    def __init__(self, attribute: Identifier, container: Expression):
        super().__init__()
        self.attribute = attribute
        self.container = container

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_attribute_access(self)


class ComparisonExpr(Expression):
    def __init__(self, lhs: Expression, op: str, rhs: Expression):
        super().__init__()
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_comparison_expr(self)

    def with_arg(self, arg: Expression, idx: int):
        assert 0 <= idx <= 1
        return ComparisonExpr(arg if idx == 0 else self.lhs, self.op, arg if idx == 1 else self.rhs)

    def negate(self):
        # TODO: reinstate as negate if ! doesn't work
        opposite = COMPARISON_NEGATIONS.get(self.op.name.lower()) if is_math_name(self.op) else None
        if opposite:
            result = ComparisonExpr(self.lhs, opposite, self.rhs)
            result.doc_string = f'Negation of {self.doc_string}'
            result.type = M_BOOLEAN
            return result
        return super().negate()


class TemporalExpr(Expression):
    def __init__(self, lhs: Expression, op: str, rhs: Expression, container: Expression):
        super().__init__()
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.container = container

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_temporal_expr(self)


class SetMembershipExpr(Expression):
    def __init__(self, expr: Expression, container: Expression):
        super().__init__()
        self.element = expr
        self.container = container

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_set_membership(self)


class NegatedExpr(Expression):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_negation(self)


class BetweenExpr(Expression):
    def __init__(self, lb, expr, ub):
        super().__init__()
        self.lb = lb
        self.expr = expr
        self.ub = ub

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_between(self)


class ComprehensionCode(Expression, ABC):
    def __init__(self, rest: 'ComprehensionCode' = None):
        super().__init__()
        self.rest = rest


class ComprehensionContainerCode(ComprehensionCode):
    def __init__(self, vars: Sequence[QualifiedName], container: Expression,
                 rest: ComprehensionCode = None):
        super().__init__(rest)
        self.vars = vars
        self.container = container

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_comprehension_container_code(self)


class ComprehensionConditionCode(ComprehensionCode):
    def __init__(self, condition: Expression, rest: ComprehensionCode = None):
        super().__init__(rest)
        self.condition = condition

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_comprehension_condition_code(self)


class StreamExpr(Expression):
    def __init__(self, term: Expression, container: ComprehensionContainerCode):
        super().__init__()
        self.term = term
        self.container = container

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_stream_expr(self)

    def as_set(self):
        aggr = AggregateExpr('SET', self.term, self.container)
        aggr.prog_free_vars = self.prog_free_vars
        return aggr


class AggregateExpr(Expression):
    def __init__(self, operator: str, term: Expression, container: ComprehensionContainerCode = None):
        super().__init__()
        self.operator = operator
        self.term = term
        self.container = container

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_aggregate_expr(self)


class LambdaExpressionExpr(Expression):
    def __init__(self, vars, body):
        super().__init__()
        self.vars = vars
        self.body = body

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_lambda_expression_expr(self)


class FunctionApplExpr(Expression):
    def __init__(self, function: QualifiedName, args, method_target: Optional[Expression] = None, named_args=(),
                 constructor=False, comment: Optional[str] = None):
        assert isinstance(function, QualifiedName)  # DEBUG
        super().__init__(comment)
        self.function = function
        self.args = args
        self.method_target = method_target
        self.constructor = constructor
        self.named_args = named_args

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_function_appl_expr(self)

    def with_function(self, function: QualifiedName) -> FunctionApplExpr:
        result = FunctionApplExpr(function, self.args, method_target=self.method_target, named_args=self.named_args)
        result.type = self.type
        result.doc_string = self.doc_string
        result.prog_free_vars = self.prog_free_vars
        return result

    def with_arg(self, arg: Expression, idx: int) -> FunctionApplExpr:
        args = copy(self.args)
        args[idx] = arg
        result = FunctionApplExpr(self.function, args, method_target=self.method_target, named_args=self.named_args)
        result.type = self.type
        result.doc_string = self.doc_string
        result.prog_free_vars = self.prog_free_vars
        return result

    def with_args(self, args: Sequence[Expression]) -> FunctionApplExpr:
        result = FunctionApplExpr(self.function, args, method_target=self.method_target, named_args=self.named_args)
        result.type = self.type
        result.doc_string = self.doc_string
        result.prog_free_vars = self.prog_free_vars
        return result


class Slice(CodeElement):
    def __init__(self, start: Expression, stop: Expression, step: Optional[Expression] = None):
        super().__init__()
        # FIXME: implement
        raise Exception('Implement me!')

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_slice(self)


class SubscriptedExpr(Expression):
    def __init__(self, obj: Expression, subscripts: Sequence[Union[Expression, Slice]]):
        super().__init__()
        self.obj = obj
        self.subscripts = subscripts

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_subscripted_expr(self)


class ConditionalExpr(Expression):
    def __init__(self, cond: Expression, pos: Expression, neg: Expression):
        super().__init__()
        self.cond = cond
        self.pos = pos
        self.neg = neg

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_conditional_expr(self)


class QuantifierExpr(Expression):
    def __init__(self, kind, formula: Expression, container: ComprehensionContainerCode,
                 unique: bool = False):
        super().__init__()
        self.kind = kind
        self.formula = formula
        self.container = container
        self.unique = unique

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_quantifier(self)


class LogicalExpr(Expression):
    def __init__(self, op: str, elements: Sequence[Expression]):
        super().__init__()
        self.op = op
        self.elements = elements

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_logical_expr(self)


class PredicateApplExpr(Expression):
    def __init__(self, pred, args):
        super().__init__()
        self.pred = pred
        self.args = args

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_predicate_appl(self)


class Concatenation(CodeElement):
    def __init__(self, *fragments):
        super().__init__()
        self.fragments = fragments

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_concatenation(self)


class LoopIn(CodeElement):
    # TODO: not in use, should be Expression returning stream?
    def __init__(self, var, container, body):
        super().__init__()
        self.var = var
        self.container = container
        self.body = body

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_loop_in(self)


class CellsExpr(Expression):
    def __init__(self, start_cell, end_cell=None):
        super().__init__()
        self.start_cell = start_cell
        self.end_cell = end_cell

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_cells(self)


def add_return_if_needed(body: CodeElement):
    """
    Add a return to the appropriate part of the body, unless it is imperative and doesn't return a value
    """
    if isinstance(body, Statements):
        return Statements(*chain(body.statements[:-1]), add_return_if_needed(body.statements[-1]))
    if isinstance(body, Imperative):
        return body
    return ReturnStatement(body)


class AbstractFunctionDefinition(CodeElement):
    def __init__(self, name: QualifiedName, typed_parameters: Sequence[ParameterDescriptor], return_type: MType,
                 body: CodeElement, func_doc_string=None, method_target=None,
                 defs: Sequence[Union['AbstractFunctionDefinition', 'AbstractClassDefinition']] = (),
                 decorators: Sequence[Expression] = ()):
        super().__init__()
        self.name = name
        self.typed_parameters = typed_parameters
        self.method_target = method_target
        self.return_type = return_type
        self.body = body
        self.func_doc_string = func_doc_string
        self.defs = defs
        self.decorators = decorators

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_abstract_function_definition(self)


class AbstractClassDefinition(CodeElement):
    def __init__(self, name: QualifiedName, superclasses: Sequence[QualifiedName], class_doc_string: str,
                 defs: ABSTRACT_DEFS_TYPE = (), decorators: Sequence[Expression] = ()):
        super().__init__()
        self.name = name
        self.superclasses = superclasses
        self.class_doc_string = class_doc_string
        self.defs = defs
        self.decorators = decorators

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_abstract_class_definition(self)


class AbstractTypeDeclaration(CodeElement):
    def __init__(self, var: str, type: MType, init: Expression = None):
        super().__init__()
        self.var = var
        self.type = type
        self.init = init

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_abstrat_type_declaration(self)


class AbstractNamedArg(Expression):
    def __init__(self, name, expr: Expression):
        super().__init__()
        self.name = name
        self.expr = expr

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_named_arg(self)


class CompilationUnit(CodeElement):
    def __init__(self, stmts: Sequence[CodeElement]):
        super().__init__()
        self.stmts = stmts
        self.type = M_ANY

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_compilation_unit(self)


class RangeCode(Expression):
    def __init__(self, start: int, stop: int):
        super().__init__()
        self.start = start
        self.stop = stop

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_range_code(self)


class Imperative(CodeElement, ABC):
    """
    Superclass for imperative (state changing) code elements
    """


class Assignment(Imperative):
    @disown('var_name')
    def __init__(self, var_name: QualifiedName, value: Expression, type_to_declare: MType = None,
                 modifiers: Sequence[str] = (), comment: Optional[str] = None):
        if comment is None and value.comment is not None:
            comment = value.comment
        super().__init__(comment)
        self.var_name = var_name
        self.value = value
        self.type_to_declare = type_to_declare
        self.modifiers = modifiers

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_assignment(self)


class ReturnStatement(Imperative):
    def __init__(self, value: Expression, comment: Optional[str] = None):
        if comment is None and value.comment is not None:
            comment = value.comment
        super().__init__(comment)
        self.value = value
        try:
            self.doc_string = value.doc_string
        except AttributeError:
            pass
        # self.free_vars = value.free_vars

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_return_statement(self)


class Statements(Imperative):
    def __init__(self, *statements: CodeElement, comment: Optional[str] = None):
        super().__init__(comment)
        self.statements = statements

    def accept(self, visitor: CodeVisitor):
        return visitor.visit_statements(self)
