from abc import ABC, abstractmethod
from numbers import Number
from typing import Tuple, AbstractSet

from codegen.optimization_analyzer import OptimizationProblemAnalyzer
from math_rep.expr import Term, Aggregate, FormalContent, MathVariableArray, MathVariable, \
    ComprehensionContainer, RangeExpr, GeneralSet, TRUE_AS_QUANTITY, FALSE_AS_QUANTITY, Quantity, TypeExpr
from math_rep.expression_types import MClassType, MAtomicType, MMappingType, QualifiedName, M_BOOLEAN, MRange, \
    is_subtype, M_NUMBER


class AbstractVariableTransformation:
    """
    An abstract class for transformations from one form of a variable to another.

    Supports transformations of expressions using the variable.

    Objects of this class receive a variable (and possibly other information) in the constructor, and are responsible
    for installing themselves on the variable.
    """

    def __init__(self):
        if type(self) is AbstractVariableTransformation:
            raise Exception('AbstractVariableTransformation is abstract and must not be instantiated')


class VariableTransformation(AbstractVariableTransformation, ABC):
    # noinspection PyMissingConstructor
    def __init__(self, var: Term):
        # FIXME: add transformer to a subclass of DomainInfo, to be put on MV and MVAs
        var.appl_info.transformer = self

    @abstractmethod
    def value_expr(self, subscripts: Tuple[Term, ...]) -> Term:
        """
        An expression that gives the value of the variable in terms of the target
        """
        raise NotImplementedError

    def eq_const_expr(self, subscripts: Tuple[Term, ...], const: Term) -> Term:
        """
        An expression that gives the value of the equality of the variable to `const` in terms of the target.

        The term ``const`` must represent a constant (but not necessarily a Quantity).

        Default implementation uses ``self.value_expr()``.
        """
        return self.value_expr(subscripts) == const

    def eq_expr_expr(self, subscripts: Tuple[Term, ...], expr: Term) -> Term:
        """
        An expression that gives the value of the equality of the variable to ``expr`` in terms of the target.

        Default implementation uses ``self.value_expr()``.
        """
        return self.value_expr(subscripts) == expr


class SingleVariableTransformation(VariableTransformation):
    """
    A transformation of a single (non-subscriptable) variable.
    """

    def __init__(self, var: Term, target_var: Term):
        super().__init__(var)
        self.target_var = target_var

    def value_expr(self, subscripts: Tuple[Term, ...]) -> Term:
        assert not subscripts
        return self.target_var

    def eq_const_expr(self, subscripts: Tuple[Term, ...], const: Term) -> Term:
        assert not subscripts
        return self.target_var == const

    def eq_expr_expr(self, subscripts: Tuple[Term, ...], expr: Term) -> Term:
        assert not subscripts
        return self.target_var == expr


class IdentityTransformation(VariableTransformation):
    def __init__(self, var: Term):
        super().__init__(var)
        self.target_var = var

    def value_expr(self, subscripts: Tuple[Term, ...]) -> Term:
        if subscripts:
            assert isinstance(self.target_var, MathVariableArray)
            return self.target_var[subscripts]
        return self.target_var


def type_to_term(mtype):
    # FIXME! make all dims of MArray be MType's
    """
    Translate an MType into a term that specifies the set of elements of that type (suitable for a container of a
    ComprehensionContainer.
    """
    if isinstance(mtype, range):
        assert mtype.step == 1
        return RangeExpr(mtype.start, mtype.stop)
    if isinstance(mtype, MRange):
        return RangeExpr(mtype.start, mtype.stop)
    if mtype == M_BOOLEAN:
        return GeneralSet([TRUE_AS_QUANTITY, FALSE_AS_QUANTITY])
    if isinstance(mtype, AbstractSet):
        if not (all(isinstance(x, Number) for x in mtype) or all(isinstance(x, str) for x in mtype)):
            raise Exception(f'Elements of domain {mtype} must all be strings or all numbers')
        return GeneralSet([Quantity(x) for x in mtype])
    raise Exception(f'No term for {mtype}')


class SimpleMappingToBinaryTransformation(VariableTransformation):
    """
    A transformation from a mapping whose range is a numeric.

    The range must NOT be boolean!

    For non-numeric range types, first convert categorical to numeric.
    """

    def __init__(self, var: MathVariableArray):
        super().__init__(var)
        vtype = var.appl_info.type
        assert isinstance(vtype, MMappingType)
        self.range_type = range_type = vtype.element_type
        assert isinstance(range_type, MAtomicType)
        assert is_subtype(range_type, M_NUMBER)
        assert range_type != M_BOOLEAN
        assert isinstance(var, MathVariableArray)
        new_var_name = FormalContent.fresh_name(var.name.with_extended_name('b').with_type(M_BOOLEAN), 'b')
        self.target_var = target_var = MathVariableArray(new_var_name, *var.dims, TypeExpr(range_type))
        # FIXME! use this when transforming expressions
        target_var.total_mapping_range_index = len(var.dims) if vtype.total else -1

    def value_expr(self, subscripts: Tuple[Term, ...]) -> Term:
        range_var_name = FormalContent.fresh_name(QualifiedName('r', self.range_type), 'r')
        range_var = MathVariable(range_var_name)
        return Aggregate('+', range_var * self.target_var[*subscripts, range_var],
                         ComprehensionContainer([range_var_name], type_to_term(self.range_type)))

    def eq_const_expr(self, subscripts: Tuple[Term, ...], const: Term) -> Term:
        return self.target_var[*subscripts, const]


class CompoundMappingToBinaryTransformation(AbstractVariableTransformation):
    """
    A transformation from a mapping whose range is a record or tuple type into separate variables for each element of
    the range.

    This transformation should be followed by the individual transformations for each reference to a field of the
    original variable.
    """

    # FIXME! if necessary, add another array (common to all the range elements) to indicate whether value is defined
    # noinspection PyMissingConstructor
    def __init__(self, var: MathVariableArray, analyzer: OptimizationProblemAnalyzer):
        var.appl_info.transformer = self
        vtype = var.appl_info.type
        assert isinstance(vtype, MMappingType)
        range_type = vtype.element_type
        # FIXME: handle tuples as well
        assert isinstance(range_type, MClassType)
        range_struct = analyzer.struct_by_name(range_type.class_name)
        self.individual_vars = {
            fname: MathVariableArray(
                FormalContent.fresh_name(var.name
                                         .with_extended_name(fname)
                                         .with_type(MMappingType(vtype.key_type, field.type, vtype.total)),
                                         'f'))
            for fname, field in range_struct.fields.items()}
        # FIXME!!! create transformations for new vars

    def field_expr(self, subscripts: Tuple[Term, ...], field: str) -> Term:
        return self.individual_vars[field][subscripts]
