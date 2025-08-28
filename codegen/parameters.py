from dataclasses import dataclass, field

from math_rep.expression_types import MType


@dataclass
class ParmKind:
    kind: str


ONLY_POS = ParmKind('pos')  # in Python only since 3.8, grammar is for 3.6
ONLY_KW = ParmKind('kw')
POS_AND_KW = ParmKind('pos+kw')
REST_PARM = ParmKind('special-rest')
KEYWORDS_PARM = ParmKind('special-keywords')
TARGET_PARM = ParmKind('method-target')
SPECIAL_PARMS = (REST_PARM, KEYWORDS_PARM)


@dataclass
class ParameterDescriptor:
    name: str
    type: MType
    kind: ParmKind = field(default_factory=lambda: POS_AND_KW)

    def for_python(self, add_type=True):
        result = self.name
        if add_type and (t := self.type) and (pt := t.for_python()):
            result += f': {pt}'
        return result
