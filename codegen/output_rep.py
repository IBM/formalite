from collections import Sequence
from numbers import Number
from typing import Optional, Union

from docplex.mp.model import Model

from codegen.utils import visitor_for
from math_rep.expression_types import QualifiedName, MType


class CodeElementOut:
    pass


class ExpressionOut(CodeElementOut):
    pass


class IdentifierOut(ExpressionOut):
    pass


class IdentifierOutOut(ExpressionOut):
    def __init__(self, name: QualifiedName):
        self.name = name


class NumberExprOut(ExpressionOut):
    def __init__(self, value: Number):
        self.value = value


class BooleanExprOut(ExpressionOut):
    def __init__(self, value: bool):
        self.value = value


class StringExprOut(ExpressionOut):
    def __init__(self, s: str):
        self.s = s


class PeriodExprOut(ExpressionOut):
    def __init__(self, start: ExpressionOut, end: ExpressionOut):
        self.start = start
        self.end = end


class TimeExprOut(ExpressionOut):
    def __init__(self, time: str):
        self.time = time


class SetExprOut(ExpressionOut):
    def __init__(self, s: Sequence[ExpressionOut]):
        self.s = s


class VariableAccessOut(ExpressionOut):
    def __init__(self, name: QualifiedName, type_name: MType = None):
        self.name = name
        self.type_name = type_name


class AttributeAccessOut(ExpressionOut):
    def __init__(self, attribute: IdentifierOut, container: ExpressionOut):
        self.attribute = attribute
        self.container = container


class ComparisonExprOut(ExpressionOut):
    def __init__(self, lhs: ExpressionOut, op: str, rhs: ExpressionOut):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs


class TemporalExprOut(ExpressionOut):
    def __init__(self, lhs: ExpressionOut, op: str, rhs: ExpressionOut, container: ExpressionOut):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.container = container


class SetMembershipExprOut(ExpressionOut):
    def __init__(self, expr: ExpressionOut, container: ExpressionOut):
        self.expr = expr
        self.container = container


class NegatedExprOut(ExpressionOut):
    def __init__(self, expr):
        self.expr = expr


class BetweenExprOut(ExpressionOut):
    def __init__(self, lb, expr, ub):
        self.lb = lb
        self.expr = expr
        self.ub = ub


class ComprehensionCodeOut:
    pass


class ComprehensionContainerCodeOut(ComprehensionCodeOut):
    def __init__(self, vars: Sequence[QualifiedName], container: ExpressionOut,
                 rest: ComprehensionCodeOut = None):
        self.vars = vars
        self.container = container
        self.rest = rest


class ComprehensionConditionCodeOut(ComprehensionCodeOut):
    def __init__(self, condition: ExpressionOut, rest: ComprehensionCodeOut = None):
        self.condition = condition
        self.rest = rest


class StreamExprOut(ExpressionOut):
    def __init__(self, term: ExpressionOut, container: ComprehensionContainerCodeOut):
        self.term = term
        self.container = container


class AggregateExprOut(ExpressionOut):
    def __init__(self, operator: str, term: ExpressionOut, container: ComprehensionContainerCodeOut = None):
        self.operator = operator
        self.term = term
        self.container = container


class FunctionApplExprOut(ExpressionOut):
    def __init__(self, function: QualifiedName, args, method_target: Optional[ExpressionOut] = None, named_args=()):
        self.function = function
        self.args = args
        self.method_target = method_target
        self.named_args = named_args


class SliceOut(CodeElementOut):
    def __init__(self, start: ExpressionOut, stop: ExpressionOut, step: Optional[ExpressionOut] = None):
        self.start = start
        self.stop = stop
        self.step = step


class SubscriptedExprOut(ExpressionOut):
    def __init__(self, obj: ExpressionOut, subscripts: Sequence[Union[ExpressionOut, SliceOut]]):
        self.obj = obj
        self.subscripts = subscripts


class ConditionalExprOut(ExpressionOut):
    def __init__(self, cond: ExpressionOut, pos: ExpressionOut, neg: ExpressionOut):
        self.cond = cond
        self.pos = pos
        self.neg = neg


class QuantifierExprOut(ExpressionOut):
    def __init__(self, kind, formula: ExpressionOut, container: ComprehensionContainerCodeOut,
                 unique: bool = False):
        self.kind = kind
        self.formula = formula
        self.container = container
        self.unique = unique


class LogicalExprOut(ExpressionOut):
    def __init__(self, op: str, elements: Sequence[ExpressionOut]):
        self.op = op
        self.elements = elements


class PredicateApplExprOut(ExpressionOut):
    def __init__(self, pred, args):
        self.pred = pred
        self.args = args


class ConcatenationOut(CodeElementOut):
    def __init__(self, *fragments):
        self.fragments = fragments


class LoopInOut(CodeElementOut):
    def __init__(self, var, container, body):
        self.var = var
        self.container = container
        self.body = body


class CellsExprOut(ExpressionOut):
    def __init__(self, start_cell, end_cell=None):
        self.start_cell = start_cell
        self.end_cell = end_cell


class ReturnStmtOut(CodeElementOut):
    def __init__(self, result: ExpressionOut):
        self.result = result


class ParameterDescriptorOut:
    pass


class AbstractFunctionDefinitionOut(CodeElementOut):
    def __init__(self, name: QualifiedName, typed_parameters: Sequence[ParameterDescriptorOut], return_type: MType,
                 body: CodeElementOut, func_doc_string=None, method_target=None,
                 defs: Sequence[Union['AbstractFunctionDefinitionOut', 'AbstractClassDefinitionOut']] = (),
                 decorators: Sequence[ExpressionOut] = ()):
        self.name = name
        self.typed_parameters = typed_parameters
        self.return_type = return_type
        self.body = body
        self.func_doc_string = func_doc_string
        self.method_target = method_target
        self.defs = defs
        self.decorators = decorators


ABSTRACT_DEFS_TYPE_OUT = Sequence[
    Union['AbstractFunctionDefinitionOut', 'AbstractClassDefinitionOut', 'AbstractTypeDeclarationOut']]


class AbstractClassDefinitionOut(CodeElementOut):
    def __init__(self, name: QualifiedName, superclasses: Sequence[QualifiedName], class_doc_string: str,
                 defs: ABSTRACT_DEFS_TYPE_OUT = (), decorators: Sequence[ExpressionOut] = ()):
        self.name = name
        self.superclasses = superclasses
        self.class_doc_string = class_doc_string
        self.defs = defs
        self.decorators = decorators


class AbstractTypeDeclarationOut(CodeElementOut):
    def __init__(self, var: str, type: MType, init: ExpressionOut = None):
        self.var = var
        self.type = type
        self.init = init


class AbstractNamedArgOut(ExpressionOut):
    def __init__(self, name, expr: ExpressionOut):
        self.name = name
        self.expr = expr


class CompilationUnitOut(CodeElementOut):
    def __init__(self, stmts: Sequence[CodeElementOut]):
        self.stmts = stmts


@visitor_for(CodeElementOut)
class OutputRepVisitor:
    def __init__(self, model: Model = None):
        self.model = model if model is not None else Model()
        self.vars = {}

    def visit_variable_access(self, var: VariableAccessOut):
        return self.vars.get(var.name) or self.model.get_var_by_name(var.name)
