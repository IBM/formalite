from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from typing import Union

from math_rep.expr import Term, ApplicationSpecificInfo, intern_expr, FormalContent


class AbstractDFlowBuilder(ABC):
    @abstractmethod
    def visit(self, expr: FormalContent):
        pass


class AbstractDFlowTable(ABC):
    def __init__(self, builder: AbstractDFlowBuilder):
        self.builder = builder
        self.agenda = OrderedDict()
        for cls in intern_expr.interned_classes:
            if issubclass(cls, FormalContent):
                cls.reset_intern_memory()

    def build(self, term: FormalContent):
        self.builder.visit(term)
        return term

    @abstractmethod
    def add_to_agenda(self, expr: FormalContent):
        pass


class DataFlow(ABC):
    def __init__(self, source: Term, target: Union[Term, ApplicationSpecificInfo], domain_table: AbstractDFlowTable):
        self.source = source
        self.target = target
        source_info = source.appl_info
        source_info.propagator_additions[id(self)] = self
        domain_table.add_to_agenda(source_info)

    @abstractmethod
    def propagate(self, domain_table: AbstractDFlowTable):
        raise NotImplementedError()

    def with_source(self, source: Term):
        result = copy(self)
        result.source = source
        return result

    def deactivate(self):
        """
        Mark this propagator for removal from its source term
        """
        self.source.appl_info.deactivated.add(id(self))


class DomainInfoWithDataFlow:
    """
    Mixin for ``AbstractDomainInfo`` that adds support for data-flow propagators
    """
    def __init__(self, *args, **kwargs):
        # initialize other superclasses
        super().__init__(*args, **kwargs)
        self.propagators = OrderedDict()
        self.propagator_additions = OrderedDict()
        self.deactivated = set()
