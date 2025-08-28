from math_rep.constants import AND_SYMBOL, FOR_ALL_SYMBOL
from math_rep.expr import LogicalOperator, Quantifier, ComprehensionContainer, \
    ComprehensionCondition
from rewriting.patterns import Bindings, let, ClassPattern
from rewriting.rules import RewriteRule, get_pre_and_post


def example1(s1, s2):
    return all(a is not None and all(b is not None and b != a for b in s2) for a in s1)


def target1_intermediate(s1, s2):
    return (all(a is not None for a in s1) and
            all(b is not None for b in s2 for a in s1) and
            all(b != a for b in s2 for a in s1))


def target1(s1, s2):
    return (all(a is not None for a in s1) and
            all(b is not None for b in s2) and
            all(b != a for b in s2 for a in s1))


class RaiseConjunction(RewriteRule):
    pattern = LogicalOperator[AND_SYMBOL,
                              let(pre=...),
                              let(arg=LogicalOperator[AND_SYMBOL, let(subargs=...)]),
                              let(post=...)]

    def transform_single(self, obj, bindings: Bindings):
        pre, post = get_pre_and_post(bindings)
        return LogicalOperator(AND_SYMBOL, [*pre, *bindings['subargs'], *post])


class RaiseConjunctionOverUniversal(RewriteRule):
    pattern = Quantifier[FOR_ALL_SYMBOL, LogicalOperator[AND_SYMBOL, let(conjuncts=...)], 'container']

    def transform_single(self, obj, bindings: Bindings):
        container = bindings['container']
        return LogicalOperator(AND_SYMBOL, [Quantifier(FOR_ALL_SYMBOL, c, container) for c in bindings['conjuncts']])


class EliminateRedundantUniversal(RewriteRule):
    pattern = Quantifier[FOR_ALL_SYMBOL, 'formula', let(container=ClassPattern(ComprehensionContainer))]

    def condition(self, obj, bindings: Bindings) -> bool:
        formula = bindings['formula']
        container: ComprehensionContainer = bindings['container']
        bound_vars = set(container.vars)
        # FIXME! add free_vars to expr classes
        free_vars = formula.free_vars
        if rest := container.rest:
            if not isinstance(rest, ComprehensionCondition):
                return False
            if rest.rest:
                # TODO: handle nested containers
                return False
            free_vars = free_vars | rest.free_vars
        return not (bound_vars & free_vars)

    def transform_single(self, obj, bindings: Bindings):
        return


# obsolete?
class ConjunctionExtractor(RewriteRule):
    pattern = LogicalOperator[AND_SYMBOL, ..., 'arg', ...]

    def transform_single(self, obj, bindings: Bindings):
        return bindings['arg']
