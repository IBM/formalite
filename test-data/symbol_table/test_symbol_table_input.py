# from validator2solver.symbol_builtins import *
import sys
import itertools
import optimistic_symbol.symbol_explore
import math_rep.expr
from validator2solver.optimistic_factory import generate_ast_from_file
from optimistic_client.unique_assignment import assignment, UniqueAssignment
from optimistic_client.unique_solution import solution_attribute
from optimistic_client.optimization import OptimizationProblem
from typing import Sequence

print(f'SYS PATH {sys.path}')
print(f'local path {__path__}')

@assignment(resource='student', activity='room', as_data_class=True)
class ASolution:
    student: str
    room: str



math_rep.expr.IFTE().describe()
math_rep.expr.Negation().to_code_rep()
math_rep.expr.Negation().to_code_rep().accept('a')
math_rep.expr.AGGREGATE_SYMBOL_MAP

@solution_attribute('second_type', as_data_class=True)
class SecondProblem(OptimizationProblem, UniqueAssignment):
    second_type: Sequence[ASolution]
    # optimistic_symbol = "aaaa"
    def zero(self):
        return optimistic_symbol.symbol_explore.parse_file('what ever file')

    def first(self):
        return pow(4, 3)

    def second(self):
        return generate_ast_from_file(f'yet another file')

    def threes(self):
        return itertools.tee()

    def execute(self):
        return self.second_type[0]

    def one_more(self):
        return self.unique_assignment()



def outer(aa):
    def inner(cc):
        return aa + cc
    return inner

# outer(4).inner(7)
