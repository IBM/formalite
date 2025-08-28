from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from itertools import chain
from operator import attrgetter
from typing import Tuple, Iterable, Sequence, Any, Type, Optional, Mapping

from math_rep.expression_types import QualifiedName


class Bindings:
    def __init__(self, bindings: Optional[Mapping['PatternVariable', Any]] = None):
        self.bindings = {} if bindings is None else bindings

    def also(self, var: 'PatternVariable', value):
        if var in self.bindings:
            raise Exception(f'Variable {var} already bound')
        return Bindings({var: value, **self.bindings})

    def also_multiple(self, vars: Sequence['PatternVariable'], value):
        result = self
        for var in vars:
            result = result.also(var, value)
        return result

    def get(self, var, default=None):
        return self.bindings.get(str_to_var(var), default)

    def __getitem__(self, var):
        return self.bindings[str_to_var(var)]

    def __repr__(self):
        vars = sorted(self.bindings.keys(), key=attrgetter('name'))
        elements = (f'{var}={self.bindings[var]}' for var in vars)
        return f'{{{", ".join(elements)}}}'


class Span:
    """
    A binding of 0 or more expressions to a single variable
    """

    def __init__(self, values: Sequence):
        self.values = values

    def __new__(cls, values: Sequence):
        if len(values) == 1:
            return next(iter(values))
        return super().__new__(cls)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f'[{", ".join(repr(v) for v in self.values)}]'


class Pattern(ABC):
    @abstractmethod
    def match(self, expr, bindings=Bindings()) -> Iterable[Bindings]:
        raise NotImplementedError('Subclasses must implement "match"')

    @abstractmethod
    def heads(self):
        """
        :return: sequence of classes that are heads of patterns that can match this pattern
        """

    def spanner_variables(self):
        """
        Return variable(s) to be bound to a span, or empty sequence if this is not a spanner pattern
        """
        return ()


@dataclass(eq=True, frozen=True)
class PatternVariable(Pattern):
    name: str

    def __str__(self):
        return f'?{self.name}'

    def match(self, expr, bindings=Bindings()) -> Iterable[Bindings]:
        if (val := bindings.get(self)) is not None:
            return pattern_match(val, expr, bindings)
        return [bindings.also(self, expr)]

    def heads(self):
        return ()


class ClassPattern(Pattern):
    def __init__(self, *classes: Type):
        self.classes = classes

    def __str__(self):
        return ' or '.join(cls.__name__ for cls in self.classes)

    def match(self, expr, bindings=Bindings()) -> Iterable[Bindings]:
        return [bindings] if isinstance(expr, self.classes) else []

    def heads(self):
        return self.classes


def str_to_var(element):
    return PatternVariable(element) if isinstance(element, str) else element


def prettify_pattern_element(arg):
    return arg.__name__ if isinstance(arg, type) else str(arg)


# any falsy value means "match anything"
MATCH_ANY = None


class CompoundPattern(Pattern):
    """
    A pattern matching an object of a given class (the "head") with given arguments.

    The pattern arguments can be other patterns (including compound patterns), as well as ellipsis (...), which can
    match zero or more actual arguments, and plain strings, which are shorthand for a pattern variable of that name,
    except as noted below.  In addition, any falsy value means "match anything"; the constant MATCH_ANY is provided
    as a meaningful, if long, way to express this.

    The first argument is treated specially in case the class is defined to have an operator (has_operator == True).
    In that case, a QualifiedName should match the *value* of the operator, and is *not* interpreted as a variable;
    a string constant will match the value of the operator if the value is a QualifiedName whose name is the given
    string.
    """

    def __init__(self, head, operator, args):
        self.head = head
        self.operator = None if operator == '' else operator
        self.args = [str_to_var(a) for a in args]

    def __str__(self):
        op = f'{operator}; ' if (operator := self.operator) is not None else ''
        return f'{prettify_pattern_element(self.head)}[{op}{", ".join(prettify_pattern_element(a) for a in self.args)}]'

    def match_operator(self, operator, bindings) -> Iterable[Bindings]:
        pattern_operator = self.operator
        if not pattern_operator:
            return [bindings]
        if isinstance(pattern_operator, Pattern):
            return pattern_match(pattern_operator, operator, bindings)
        if isinstance(pattern_operator, str) and isinstance(operator, QualifiedName):
            operator = operator.name
        if pattern_operator == operator:
            return [bindings]
        return []

    def match_operands(self, remaining_patterns, remaining_operands, bindings) -> Iterable[Bindings]:
        if not remaining_patterns:
            return [bindings]
        pat0 = remaining_patterns[0]
        vars = ()
        if pat0 is ... or isinstance(pat0, Pattern) and (vars := pat0.spanner_variables()):
            if len(remaining_patterns) == 1:
                # ... in last position, make greedy
                return [bindings.also_multiple(vars, Span(remaining_operands))]
            result = chain.from_iterable(
                self.match_operands(remaining_patterns[1:], remaining_operands[i:],
                                    bindings.also_multiple(vars, Span(remaining_operands[:i])))
                for i in range(len(remaining_operands)))
            return result
        if not remaining_operands:
            return []
        return chain.from_iterable(self.match_operands(remaining_patterns[1:], remaining_operands[1:], new_bindings)
                                   for new_bindings in
                                   pattern_match(pat0, remaining_operands[0], bindings))

    def match(self, expr, bindings=Bindings()) -> Iterable[Bindings]:
        if not isinstance(expr, self.head):
            return []
        if expr.has_operator:
            new_bindings = self.match_operator(expr.operator(), bindings)
        else:
            new_bindings = [bindings]
        return chain.from_iterable(self.match_operands(self.args, expr.arguments(), b) for b in new_bindings)

    def heads(self):
        return self.head,


class Let(Pattern):
    def __init__(self, var: PatternVariable, pattern: Pattern):
        self.var = str_to_var(var)
        self.pattern = str_to_var(pattern)

    def __str__(self):
        return f'{self.var}={prettify_pattern_element(self.pattern)}'

    def match(self, expr, bindings=Bindings()) -> Iterable[Bindings]:
        if bindings.get(self.var) is not None:
            raise Exception(f'Variable {self.var} already bound in Let')
        return (new_bindings.also(self.var, expr) for new_bindings in pattern_match(self.pattern, expr, bindings))

    def heads(self):
        return self.pattern.heads()

    def spanner_variables(self):
        expr_vars = self.pattern.spanner_variables() if isinstance(self.pattern, Pattern) else ()
        return expr_vars + (self.var,) if expr_vars or self.pattern is ... else ()


class OneOf(Pattern):
    """
    A pattern that matches any of its arguments.
    """

    def __init__(self, *values):
        self.values = values

    def match(self, expr, bindings=Bindings()) -> Iterable[Bindings]:
        return [bindings] if expr in self.values else []

    def heads(self):
        return frozenset()


def one_of(*values):
    return OneOf(*values)


class OrPattern(Pattern):
    def __init__(self, *patterns: Pattern):
        self.patterns = patterns

    def match(self, expr, bindings=Bindings()) -> Iterable[Bindings]:
        for pattern in self.patterns:
            match = pattern.match(expr, bindings)
            first = next(match, None)
            if first is None:
                return iter([])
            return chain([first], match)

    def heads(self):
        return frozenset(cls for pat in self.patterns for cls in pat.heads())

    def spanner_variables(self):
        return frozenset(chain.from_iterable(pat.spanner_variables() for pat in self.patterns))


def let(**bindings: Pattern):
    assert len(bindings) == 1
    for var, pattern in bindings.items():
        return Let(PatternVariable(var), pattern)


class MetaPatternable(ABCMeta):
    def __getitem__(self, item) -> Pattern:
        items = item if isinstance(item, Tuple) else (item,)
        if hasattr(self, 'has_operator') and self.has_operator:
            operator = items[0]
            args = items[1:]
        else:
            operator = None
            args = items
        return CompoundPattern(self, operator, args)


def pattern_match(pattern, obj, bindings: Bindings = Bindings()) -> Iterable[Bindings]:
    if not pattern or pattern is ...:
        return iter([bindings])
    if isinstance(pattern, Pattern):
        return pattern.match(obj, bindings)
    if isinstance(pattern, type) and isinstance(obj, pattern):
        return iter([bindings])
    if obj == pattern:
        return iter([bindings])
    return iter([])


def find_recursive_matches(pattern, obj, bindings: Bindings = Bindings()) -> Iterable[Tuple[Any, Bindings]]:
    yield from ((obj, binding) for binding in pattern_match(pattern, obj, bindings))
    for arg in obj.arguments():
        yield from find_recursive_matches(pattern, arg, bindings)
