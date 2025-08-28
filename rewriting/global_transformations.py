from typing import Mapping, Sequence

from math_rep.expr import Term, MathVariable, FormalContent, Condition, Comparison, Quantity
from math_rep.expression_types import QualifiedName
from math_rep.math_symbols import EQ_QN
from rewriting.patterns import ClassPattern, Bindings, let, OrPattern
from rewriting.rules import RewriteRule, RuleSet, OrderedRuleSets, exhaustively_apply_rules


class SubstitutionRule(RewriteRule):
    def __init__(self, substitutions: Mapping[QualifiedName, Term]):
        self.substitutions = substitutions

    pattern = ClassPattern(MathVariable)

    def condition(self, obj: MathVariable, bindings: Bindings) -> bool:
        return obj.name in self.substitutions

    def transform_single(self, obj: MathVariable, bindings: Bindings):
        return self.substitutions[obj.name]


def substitute_globally(exprs: Sequence[FormalContent], substitutions: Mapping[QualifiedName, Term]
                        ) -> Sequence[FormalContent]:
    rules = OrderedRuleSets(RuleSet(SubstitutionRule(substitutions)))
    return [exhaustively_apply_rules(rules, expr) for expr in exprs]


def find_constants(constraints: Sequence[Condition]):
    pat = OrPattern(Comparison[EQ_QN, let(var=MathVariable), let(val=Quantity)],
                    Comparison[EQ_QN, let(val=Quantity), let(var=MathVariable)])
    return {m['var'].name: m['val']
            for constraint in constraints
            if (matches := pat.match(constraint)) and (m := next(matches, None)) is not None}
