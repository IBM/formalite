from collections import defaultdict
from itertools import chain
from typing import Union

from math_rep.dataflow import AbstractDFlowTable
from rewriting.patterns import Pattern, Bindings, pattern_match, Span


class RewriteRule:
    @property
    def pattern(self) -> Pattern:
        raise NotImplementedError()

    def condition(self, obj, bindings: Bindings, *pargs, **kwargs) -> bool:
        return True

    def transform_single(self, obj, bindings: Bindings, *pargs, **kwargs):
        return None

    def transform(self, obj, bindings: Bindings, *pargs, **kwargs):
        if (result := self.transform_single(obj, bindings, *pargs, **kwargs)) is not None:
            return [result]
        raise NotImplementedError('Must implement "transform", or "transform_single" must return a value')


def all_subclasses(cls):
    return chain([cls], chain.from_iterable(all_subclasses(sub) for sub in cls.__subclasses__()))


DEFAULT_SUBSTITUTION_RULES = []
DEFAULT_EQUALITY_RULES = []


class RuleSet:
    def __init__(self, *rules: RewriteRule,
                 substitution_rules=DEFAULT_SUBSTITUTION_RULES,
                 equality_rules=DEFAULT_EQUALITY_RULES):
        self.substitution_rules = substitution_rules
        self.equality_rules = equality_rules
        rules = [rule for rule in rules if rule]
        self._rules = rules
        heads = defaultdict(list)
        for rule in rules:
            for cls in rule.pattern.heads():
                for sub in all_subclasses(cls):
                    heads[sub].append(rule)
        self.heads = heads

    def __iter__(self):
        return iter(self._rules)


class OrderedRuleSets:
    def __init__(self, *rulesets: Union[RuleSet, RewriteRule]):
        self._rulesets = [ruleset if isinstance(ruleset, RuleSet) else RuleSet(ruleset)
                          for ruleset in rulesets if ruleset]

    def __iter__(self):
        return iter(self._rulesets)


class SequentialRuleSets:
    def __init__(self, *orulesets: Union[OrderedRuleSets, RuleSet, RewriteRule]):
        self._orulesets = [ruleset if isinstance(ruleset, OrderedRuleSets) else
                           OrderedRuleSets(ruleset) if isinstance(ruleset, RuleSet) else
                           OrderedRuleSets(RuleSet(ruleset))
                           for ruleset in orulesets if ruleset]

    def __iter__(self):
        return iter(self._orulesets)


def apply_rule(rule: RewriteRule, obj, *pargs, **kwargs):
    bindings = next(iter(pattern_match(rule.pattern, obj)), None)
    if bindings is not None and rule.condition(obj, bindings, *pargs, **kwargs):
        result = next(iter(rule.transform(obj, bindings, *pargs, **kwargs)))
        # print(f'Rule: {rule.__class__.__name__}')
        # print(f'Obj: {obj}')
        # print(f'Transformed: {result} (Class {result.__class__.__name__})')
        return result
    else:
        return obj


def prevent_type_propagation(obj):
    obj._prevent_type_propagation = True
    return obj


def apply_rules_bottom_up(rules: RuleSet, obj, domain_table: AbstractDFlowTable = None, *pargs, **kwargs):
    # print(f'*** Trying: {str(obj)} (Class {obj.__class__.__name__})')
    given = obj
    args = obj.arguments()
    new_args = [apply_rules_bottom_up(rules, arg, domain_table=domain_table, *pargs, **kwargs) for arg in args]
    if any(na is not oa for na, oa in zip(new_args, args)):
        obj = obj.with_arguments(new_args)
        if domain_table:
            domain_table.build(obj)
            for rule in rules.substitution_rules:
                rule(given, obj, domain_table)
                rule(obj, given, domain_table)
        return obj
    for rule in rules.heads.get(obj.__class__, ()):
        obj = apply_rule(rule, obj, *pargs, **kwargs)
        if obj is not given:
            # print(f'Rule {rule.__class__.__name__} changed {given} to {obj} (Class {obj.__class__.__name__})')
            if domain_table:
                domain_table.build(obj)
                for rule in rules.equality_rules:
                    rule(given, obj, domain_table, prevent_type_propagation=hasattr(obj, '_prevent_type_propagation'))
                    rule(obj, given, domain_table, prevent_type_propagation=hasattr(obj, '_prevent_type_propagation'))
            return obj
    return obj


def exhaustively_apply_rules(seqrulesets: Union[SequentialRuleSets, OrderedRuleSets, RuleSet, RewriteRule], obj,
                             domain_table: AbstractDFlowTable = None, print_intermediate=False, *pargs, **kwargs):
    if isinstance(seqrulesets, OrderedRuleSets):
        seqrulesets = SequentialRuleSets(seqrulesets)
    elif isinstance(seqrulesets, RuleSet):
        seqrulesets = SequentialRuleSets(OrderedRuleSets(seqrulesets))
    elif isinstance(seqrulesets, RewriteRule):
        seqrulesets = SequentialRuleSets(OrderedRuleSets(RuleSet(seqrulesets)))
    for rulesets in seqrulesets:
        # FIXME! this is very inefficient!
        repeat = True
        while repeat:
            for ruleset in rulesets:
                result = apply_rules_bottom_up(ruleset, obj, domain_table=domain_table, *pargs, **kwargs)
                repeat = result is not obj
                if repeat:
                    if print_intermediate:
                        print(result)
                    obj = result
                    break
            if not repeat and domain_table and not domain_table.empty_agenda():
                domain_table.propagate()
                repeat = True
    return obj


# TODO: this applies a single rule, need code to apply to subterms;
# meanwhile using workaround of creating multiple bindings
def get_all_rule_results(rule: RewriteRule, obj, *pargs, **kwargs):
    return (result
            for bindings in (pattern_match(rule.pattern, obj))
            for result in rule.transform(obj, bindings, *pargs, **kwargs))


def get_pre_and_post(bindings, names=('pre', 'post')):
    pre = bindings[names[0]]
    if isinstance(pre, Span):
        pre = pre.values
    else:
        pre = [pre]
    post = bindings[names[1]]
    if isinstance(post, Span):
        post = post.values
    else:
        post = [post]
    return pre, post
