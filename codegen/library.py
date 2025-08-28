from collections import namedtuple

from codegen.abstract_rep import Expression, DummyVariable

AttributeMapping = namedtuple('AttributeMapping', ('expr', 'dummy'))


class AttributeMappings:
    def __init__(self):
        self.map = {}

    def add_mapping(self, attribute_name: str, expression: Expression, dummy_var: DummyVariable):
        self.map[attribute_name] = AttributeMapping(expression, dummy_var)

    def get_mapping(self, attribute_name: str):
        return self.map.get(attribute_name)
