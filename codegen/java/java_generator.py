from abc import ABC, abstractmethod
from itertools import chain, repeat, groupby
from numbers import Number
from operator import attrgetter
from typing import List, Union, Sequence

from codegen.abstract_rep import CodeElement, CodeFragment, CompilationUnit, FunctionDefinition, Import, \
    LogicalExpr, Expression, FunctionApplExpr, ComparisonExpr, VariableAccess, NumberExpr, \
    BooleanExpr, StringExpr, parenthesize, ReturnStatement, SubscriptedExpr, AggregateExpr, \
    ComprehensionContainerCode, ComprehensionConditionCode, NegatedExpr, ConditionalExpr, CastExpr, SetExpr, \
    LambdaExpressionExpr, Assignment, AttributeAccess
from codegen.java.java_symbols import JavaFunctionModule, JAVA_INTERNAL_FRAME_NAME, JAVA_STREAM_ITERATOR_NEXT_QU, \
    MAX_LANG_MATH_QN, MIN_LANG_MATH_QN, THIS
from codegen.transformations import ArgRef, MultipleArgs, SpliceArgs, TransformToAttribute
from codegen.utils import visitor_for
from math_rep.constants import NOT_EQUALS_SYMBOL, AND_SYMBOL, OR_SYMBOL, IMPLIES_SYMBOL, \
    INTERSECTION_SYMBOL, UNION_SYMBOL, NOT_SYMBOL, LE_SYMBOL, GE_SYMBOL, ELEMENT_OF_SYMBOL, NOT_ELEMENT_OF_SYMBOL
from math_rep.expr import function_type
from math_rep.expression_types import QualifiedName, NameTranslator, M_BOOLEAN, MArray, MFunctionType, as_math_name, \
    is_math_name
from math_rep.math_symbols import is_make_array

MAX_INT15 = (2 << 14) - 1
MIN_INT15 = -(2 << 14)
MAX_INT31 = (2 << 30) - 1
MIN_INT31 = -(2 << 30)
MAX_INT63 = (2 << 62) - 1
MIN_INT63 = -(2 << 62)

ASSOCIATIVE_PYTHON_OPERATORS = {'+', '*', 'and', 'or'}

JAVA_COMP_OPS = {'=': '==', NOT_EQUALS_SYMBOL: '!=', '>': '>', GE_SYMBOL: '>=', '<': '<', LE_SYMBOL: '<=',
                 ELEMENT_OF_SYMBOL: 'in', NOT_ELEMENT_OF_SYMBOL: 'not in'}

# Precedence table can be found at:
# https://introcs.cs.princeton.edu/java/11precedence/
# https://docs.oracle.com/javase/tutorial/java/nutsandbolts/operators.html
JAVA_OPERATOR_PRECEDENCE = {'+': 160, '-': 160, '*': 170, '/': 170, '%': 169, '|': 153, '&': 156, '==': 130, '<': 140,
                            '<=': 140, '=': 80, '>=': 140, '>': 140, '!=': 130, '&&': 110, '||': 100}
JAVA_UNARY_OPERATOR_PRECEDENCE = {'+': 180, '-': 180, '!': 180, '~': 180}
JAVA_CAST_PRECEDENCE = 175
JAVA_LAMBDA_PRECEDENCE = 10
JAVA_TERNARY_PRECEDENCE = 90
JAVA_ARRAY_ACCESS_PRECEDENCE = 190
JAVA_MEMBER_ACCESS_PRECEDENCE = 190

JAVA_AGGREGATE_COLLECTORS = {'SET': 'Collectors.toSet()'}

FUNCTION_TRANSLATIONS = {INTERSECTION_SYMBOL: '&', UNION_SYMBOL: '|', AND_SYMBOL: '&&',
                         OR_SYMBOL: '||',
                         IMPLIES_SYMBOL: (OR_SYMBOL, (NOT_SYMBOL, ArgRef(0)), ArgRef(1)),
                         # 'max': JavaFunctionModule(MAX_LANG_MATH_QN),
                         # 'min': JavaFunctionModule(MIN_LANG_MATH_QN),
                         }

JAVA_BUILTIN_FRAME = (JAVA_INTERNAL_FRAME_NAME,)


def java_builtin(name: str) -> QualifiedName:
    return QualifiedName(name, JAVA_BUILTIN_FRAME)


JAVA_FUNCTION_TRANSLATIONS = {AND_SYMBOL: java_builtin('&&'),
                              OR_SYMBOL: java_builtin('||'),
                              NOT_SYMBOL: java_builtin('!'),
                              LE_SYMBOL: java_builtin('<='),
                              GE_SYMBOL: java_builtin('>='),
                              NOT_EQUALS_SYMBOL: java_builtin('!='),
                              '=': java_builtin('=='),
                              '+': java_builtin('+'),
                              '*': java_builtin('*'),
                              '-': java_builtin('-'),
                              '/': java_builtin('/'),
                              '%': java_builtin('%'),
                              'next': QualifiedName('next', lexical_path=JAVA_STREAM_ITERATOR_NEXT_QU),
                              'min': MIN_LANG_MATH_QN,
                              'max': MAX_LANG_MATH_QN,
                              }

JAVA_BOOL_MAPPINGS = {'True': 'true', 'False': 'false'}


class JavaFunctionNameTranslator(NameTranslator):
    def translate(self, name: QualifiedName):
        if is_math_name(name):
            return JAVA_FUNCTION_TRANSLATIONS[name.name.lower()]
        return name
        # if is_python_builtin(name):
        #     return name
        # if is_python_name(name):
        #     return name
        # raise Exception(f'Unknown Python function {name}')


def apply_transformation(template, args: List[Expression], doc_string):
    if isinstance(template, Sequence):
        transformed_args = []
        for e in template[1:]:
            ta = apply_transformation(e, args, '???')
            if isinstance(ta, SpliceArgs):
                transformed_args.extend(ta.args)
            else:
                transformed_args.append(ta)
        func_name = as_math_name(template[0])
        result = FunctionApplExpr(func_name, transformed_args)
        result.type = function_type(func_name)
    elif isinstance(template, str):
        result = StringExpr(template)
    elif isinstance(template, Number):
        result = NumberExpr(template)
    elif isinstance(template, ArgRef):
        result = args[template.num]
    elif isinstance(template, ConditionalTransformation) and template.condition(args):
        return apply_transformation(template.transformation(), args, doc_string)
    elif isinstance(template, MultipleArgs):
        result = SpliceArgs(args[slice(template.start, template.stop)])
    elif isinstance(template, TransformToAttribute):
        assert len(args) == 1, 'Only one argument possible for attribute access'
        result = AttributeAccess(template.attribute, args[0])
    else:
        raise Exception(f'Unrecognized template element: {template}')
    result.doc_string = doc_string
    return result


class ConditionalTransformation(ABC):
    @abstractmethod
    def condition(self, args):
        pass

    @abstractmethod
    def transformation(self):
        pass


class TransformDifferent(ConditionalTransformation):
    def condition(self, args):
        return len(args) == 2

    def transformation(self):
        return NOT_EQUALS_SYMBOL, ArgRef(0), ArgRef(1)


@visitor_for(CodeElement, collect_results=False)
class AbstractRepVisitor:
    pass


class JavaVisitor(AbstractRepVisitor):
    def __init__(self,
                 base_name,
                 function_name_translator_class=JavaFunctionNameTranslator,
                 attribute_mappings=(),
                 use_class_name_for_static_methods=False,
                 lexical_paths_to_ignore=()):
        self.base_name = base_name
        self.imports = set()
        self.dummy_vars = {}
        self.attribute_mappings = attribute_mappings
        self.helpers = []
        self.var_counter = 0
        self.func_counter = 0
        self.function_name_translator = function_name_translator_class()
        self.use_class_name_for_static_methods = use_class_name_for_static_methods
        self.lexical_paths_to_ignore = lexical_paths_to_ignore

    def variable_to_java(self, var: QualifiedName) -> str:
        """
        Override this function to add decoration to a variable name in references and assignments
        """
        return var.to_c_identifier()

    @abstractmethod
    def pretty_function_def(self, helper: FunctionDefinition) -> str:
        pass

    def pretty_helpers(self, checkpoint=0):
        # TODO: use string appropriate for specific language
        return '\n\n'.join(self.pretty_function_def(h) for h in self.helpers[checkpoint:])

    def add_import(self, imp: Import):
        self.imports.add(imp)

    def pretty_imports(self):
        result = []
        for module, contents in groupby(sorted(self.imports, key=attrgetter('module', 'name')), attrgetter('module')):
            result.append(f'{module}: {", ".join(map(attrgetter("name"), contents))}')
        return '\n'.join(result)

    def _add_import_from_var(self, var):
        # FIXME: implement this; ignore user vars and vars from current module (need to define in __init__)
        pass

    def add_helper_function(self, func: FunctionDefinition):
        self.helpers.append(func)

    def full_code(self, abs_rep: CodeElement, paraphrase, encapsulate=True):
        # code = self.visit(add_return_if_needed(abs_rep))
        code = self.visit(abs_rep)
        imports = []
        for module, contents in groupby(sorted(self.imports, key=attrgetter('module', 'name')), attrgetter('module')):
            imports.append(f'from {module} import {", ".join(map(attrgetter("name"), contents))}')
        body = (self.encapsulate(code, additional_doc=paraphrase, checkpoint=0) if encapsulate
                else code)
        helpers = self.pretty_helpers()
        return '\n'.join(imports) + ('\n\n\n' if imports else '') + helpers + ('\n\n\n' if helpers else '') + body.value

    def encapsulate(self, cf: CodeFragment, checkpoint: int, additional_doc=None) -> CodeFragment:
        return cf

    def _check_logical_compatible(self, expr):
        if isinstance(expr, (LogicalExpr, ComparisonExpr)):
            return True
        if isinstance(expr, VariableAccess):
            if expr.name.type in (M_BOOLEAN,):
                return True
        if isinstance(expr, FunctionApplExpr):
            if expr.type in (M_BOOLEAN,):
                return True
        return False

    def _application_to_code(self, function_qn: QualifiedName, doc_string, args, named_args=None, static_method=False):
        function = function_qn
        # FIXME! translation of function depends on lexical-path, function must be a QualifiedName
        template = FUNCTION_TRANSLATIONS.get(function.name if isinstance(function, QualifiedName) else function)
        while template:
            if isinstance(template, str):
                function = QualifiedName(template, lexical_path=JAVA_BUILTIN_FRAME)
                break
            elif isinstance(template, dict):
                assert len(args) >= 1
                template = template.get(args[0].type) or template.get(None)
            elif isinstance(template, JavaFunctionModule):
                function = template.call()
                break
            else:
                transformed = apply_transformation(template, args, doc_string)
                return transformed.accept(self)

        function = self.function_name_translator.translate(function)
        function_name = function.name
        precedence = (JAVA_UNARY_OPERATOR_PRECEDENCE if len(args) == 1 else JAVA_OPERATOR_PRECEDENCE).get(function_name)
        arg_bindings = (repeat(precedence) if function_name in ASSOCIATIVE_PYTHON_OPERATORS
                        else chain([precedence], repeat((precedence or 0) + 1)))

        args_code = [a.accept(self) for a in args]
        free_vars = frozenset(chain.from_iterable(ac.free_vars for ac in args_code))
        args_text = [parenthesize(arg_precedence, ac) for ac, arg_precedence in zip(args_code, arg_bindings)]
        if precedence:
            if len(args_text) > 1:
                return CodeFragment(f' {function_name} '.join(args_text), precedence=precedence, free_vars=free_vars)
            else:
                return CodeFragment(f'{function_name}{" " if function_name[-1].isalnum() else ""}{args_text[0]}',
                                    precedence=precedence, free_vars=free_vars)
        arglist = ', '.join(args_text)
        if named_args:
            arglist += ', ' + ', '.join(f'{name}={value.accept(self)[0]}' for name, value in named_args)
        if static_method and function.lexical_path not in self.lexical_paths_to_ignore:  # special case, imported with '*'
            function_name = f'{function.lexical_path[0]}.{function_name}'
        return CodeFragment(f'{function_name}({arglist})', doc_string=doc_string, free_vars=free_vars)

    def visit_compilation_unit(self, cu: CompilationUnit):
        stmts = [self.visit(s) for s in cu.stmts]
        return CodeFragment('\n'.join(s.value for s in stmts),
                            free_vars=frozenset(chain.from_iterable(s.free_vars for s in stmts)))

    def visit_function_appl_expr(self, appl: FunctionApplExpr):
        if not appl.method_target and not appl.constructor:
            return self._application_to_code(appl.function, appl.doc_string, appl.args, appl.named_args,
                                             static_method=True)
        if appl.constructor:
            target = 'new '
        elif isinstance(appl.method_target, VariableAccess) and appl.method_target.name == THIS:
            target = ''
        else:
            target = f'{parenthesize(190, self.visit(appl.method_target))}.'
        args_code = [a.accept(self) for a in appl.args]
        free_vars = frozenset(chain.from_iterable(ac.free_vars for ac in args_code))
        # FIXME: add lexical path components as required based on imports
        if is_make_array(function := appl.function):
            function_type = appl.function.type
            assert isinstance(function_type, MFunctionType)
            result_type = function_type.result_type
            assert isinstance(result_type, MArray)
            # FIXME!! instead of for_java() use the name with an import
            array_type = f'{result_type.element_type.for_java()}{"[]" * len(result_type.dims)}'
            return CodeFragment(f'new {array_type} {{ {", ".join(a.value for a in args_code)} }}',
                                free_vars=free_vars)
        return CodeFragment(f'{target}{function.name}({", ".join(a.value for a in args_code)})',
                            free_vars=free_vars)

    def visit_attribute_access(self, attr: AttributeAccess):
        container = self.visit(attr.container)
        return CodeFragment(f'{parenthesize(JAVA_MEMBER_ACCESS_PRECEDENCE, container)}'
                            f'.{attr.attribute.name.to_c_identifier()}',
                            precedence=JAVA_MEMBER_ACCESS_PRECEDENCE,
                            free_vars=container.free_vars)

    def visit_lambda_expression_expr(self, lfunc: LambdaExpressionExpr):
        body = self.visit(lfunc.body)
        vars_desc = (lfunc.vars[0].name if len(lfunc.vars) == 1
                     else f"({', '.join(arg.name for arg in lfunc.vars)})")
        # FIXME!!!!! remove lambda bound vars from free_vars
        free_vars = body.free_vars
        return CodeFragment(f'{vars_desc} -> {body.value}', precedence=JAVA_LAMBDA_PRECEDENCE, free_vars=free_vars)

    def visit_conditional_expr(self, ifte: ConditionalExpr):
        cond_cf = self.visit(ifte.cond)
        pos_cf = self.visit(ifte.pos)
        neg_cf = self.visit(ifte.neg)
        return CodeFragment(f'{parenthesize(JAVA_TERNARY_PRECEDENCE, cond_cf)} '
                            f'? {parenthesize(JAVA_TERNARY_PRECEDENCE, pos_cf)} '
                            f': {parenthesize(JAVA_TERNARY_PRECEDENCE, neg_cf)}',
                            precedence=JAVA_TERNARY_PRECEDENCE,
                            free_vars=frozenset(cond_cf.free_vars | pos_cf.free_vars | neg_cf.free_vars))

    def visit_logical_expr(self, expr: LogicalExpr):
        code = self._application_to_code(expr.op, expr.doc_string, expr.elements)
        if not all(self._check_logical_compatible(operand) for operand in expr.elements):
            raise Exception(f'Logical expression {code.value}\n must consists of boolean operands')
        return code

    def visit_comparison_expr(self, comp: ComparisonExpr):
        py_op = JAVA_COMP_OPS[comp.op.name]
        precedence = JAVA_OPERATOR_PRECEDENCE.get(py_op)
        lhs_code = comp.lhs.accept(self)
        lhs_str = parenthesize(precedence, lhs_code)
        rhs_code = comp.rhs.accept(self)
        rhs_str = parenthesize(precedence, rhs_code)

        return CodeFragment(f'{lhs_str} {py_op} {rhs_str}', precedence=precedence,
                            free_vars=lhs_code.free_vars | rhs_code.free_vars)

    def visit_variable_access(self, variable: VariableAccess):
        var = variable.name
        self._add_import_from_var(var)
        return CodeFragment(self.variable_to_java(var), free_vars=frozenset([var]))

    def visit_number(self, num: NumberExpr):
        value = num.value
        java_rep = str(value) if not isinstance(value, int) or MIN_INT31 <= value <= MAX_INT31 else f'{value}L'
        return CodeFragment(java_rep)

    def visit_boolean(self, value: BooleanExpr):
        return CodeFragment(JAVA_BOOL_MAPPINGS.get(str(bool(value.value))))

    def visit_string(self, s: StringExpr):
        return CodeFragment(convert_to_java_string(s.value))

    def visit_subscripted_expr(self, sub: SubscriptedExpr):
        obj_cf = self.visit(sub.obj)
        subscripts_cf = [self.visit(s) for s in sub.subscripts]
        return CodeFragment(
            f'{parenthesize(JAVA_ARRAY_ACCESS_PRECEDENCE, obj_cf)}[{", ".join(s.value for s in subscripts_cf)}]',
            precedence=JAVA_ARRAY_ACCESS_PRECEDENCE,
            free_vars=obj_cf.free_vars | frozenset(
                chain.from_iterable(s.free_vars for s in subscripts_cf)))

    def visit_aggregate_expr(self, aggregate: AggregateExpr):
        term_code = aggregate.term.accept(self)
        container_code = aggregate.container.accept(self)
        bound_vars = container_code.bound_vars
        free_vars = (container_code.free_vars | term_code.free_vars) - bound_vars
        collector = JAVA_AGGREGATE_COLLECTORS[aggregate.operator]
        inside = f'{container_code.value}.map({", ".join(var.name for var in aggregate.container.vars)} -> {term_code.value})'
        code = f'{inside}.collect({collector})'
        return CodeFragment(code, free_vars=free_vars)

    def visit_comprehension_container_code(self, compr: ComprehensionContainerCode):
        rest_code, rest_vars, rest_bound_vars = self.code_from_rest(compr)
        container_code = self.visit(compr.container)
        bound_vars = set(compr.vars) | rest_bound_vars
        if rest_code:
            extension = f'{rest_code.value}({", ".join(var.name for var in compr.vars)} -> {rest_code.body})' if rest_code.body else rest_code.value
        else:
            extension = ''
        result = CodeFragment(f'{container_code.value}.stream(){"." + extension if extension else ""}',
                              free_vars=(rest_vars | container_code.free_vars) - bound_vars)
        result.bound_vars = bound_vars
        result.container_code = container_code
        return result

    def visit_comprehension_condition_code(self, compr: ComprehensionConditionCode):
        rest_code, rest_vars, rest_bound_vars = self.code_from_rest(compr)
        condition_code = self.visit(compr.condition)
        result = CodeFragment(f'filter',
                              body=condition_code.value,
                              free_vars=condition_code.free_vars | rest_vars)
        result.bound_vars = rest_bound_vars
        result.rest_code = rest_code
        return result

    def visit_set_expr(self, s: SetExpr):
        elements = [e.accept(self) for e in s.set]
        return CodeFragment('Set.of(' + ', '.join(e.value for e in elements) + ')',
                            free_vars=frozenset(chain.from_iterable(e.free_vars for e in elements)))

    def code_from_rest(self, compr: Union[ComprehensionContainerCode, ComprehensionConditionCode]):
        if (rest := compr.rest) is not None:
            rest_code = self.visit(rest)
            rest_code.value = rest_code.value
            rest_vars = rest_code.free_vars
            rest_bound_vars = rest_code.bound_vars
        else:
            rest_code = ''
            rest_vars = set()
            rest_bound_vars = set()
        return rest_code, rest_vars, rest_bound_vars

    def visit_negation(self, expr: NegatedExpr):
        expr_code = expr.expr.accept(self)
        precedence = JAVA_UNARY_OPERATOR_PRECEDENCE.get('!')
        return CodeFragment('!' + parenthesize(precedence, expr_code), precedence=precedence,
                            free_vars=expr_code.free_vars)

    def visit_cast_expr(self, cast: CastExpr):
        term_code = self.visit(cast.term)
        return CodeFragment(f'({cast.type.for_java().name}) {parenthesize(JAVA_CAST_PRECEDENCE, term_code)}',
                            precedence=JAVA_CAST_PRECEDENCE)

    def visit_assignment(self, assign: Assignment):
        value_code = assign.value.accept(self)
        modifiers = ''.join(f'{mod} ' for mod in (assign.modifiers or ()))
        decl = f'{assign.type_to_declare.for_java().name} ' if assign.type_to_declare is not None else ''
        return CodeFragment(f'{modifiers}{decl}{self.variable_to_java(assign.var_name)} = {value_code.value}',
                            precedence=JAVA_LAMBDA_PRECEDENCE,
                            free_vars=value_code.free_vars)

    def visit_return_statement(self, ret: ReturnStatement):
        value = self.visit(ret.value)
        return CodeFragment(f'return {value.value}', precedence=JAVA_LAMBDA_PRECEDENCE, free_vars=value.free_vars)


JAVA_ESCAPING_REPLACEMENTS = {
    '\\': '\\\\',
    '\n': '\\n',
    '\t': '\\t',
    '\r': '\\r',
    '\f': '\\f',
    # "'": "\\'",
    '"': '\\"'
}


def convert_to_java_string(var: str):
    for key, replace in JAVA_ESCAPING_REPLACEMENTS.items():
        var = var.replace(key, replace)
    return f'\"{var}\"'
