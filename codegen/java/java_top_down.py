from typing import Union

from codegen.java.java_symbols import JAVA_STREAM_QN, JAVA_STREAM_FILTER_QN, JAVA_STREAM_MAP_QN, \
    JAVA_STREAM_FLATMAP_QN, JAVA_STREAM_COLLECT_TO_LIST_QN, static_method_to_class_transform, JAVA_STREAM_COLLECT_QN
from math_rep.expr import Aggregate, Stream, ComprehensionContainer, ComprehensionCondition, FunctionApplication, \
    LambdaExpression, MathVariable
from math_rep.expression_types import QualifiedName


def to_qn_lambda(name):
    if isinstance(name, str):
        return QualifiedName(name, lexical_path=('*java-lambda*',))
    if isinstance(name, QualifiedName):
        return name.with_path(('*java-lambda*',), override=True)


def transform_top_down(obj):
    if isinstance(obj, (Aggregate, Stream)):
        return visit_aggregate(obj)
    return obj


def visit_aggregate(obj: Union[Aggregate, Stream], collector=True):
    term, container = obj.arguments()
    # Transform the Aggregate Term
    if isinstance(term, (Aggregate, Stream)):
        term = visit_aggregate(term)

    # Transform the Aggregate Container
    container_func = transform_container(term, container)

    # Add collector if necessary
    if collector:
        # TODO: Handle Collector Types SET/ List/
        collectors_aggregate_func = FunctionApplication(function=JAVA_STREAM_COLLECT_TO_LIST_QN,
                                                        args=[],
                                                        method_target=MathVariable(static_method_to_class_transform(
                                                            JAVA_STREAM_COLLECT_TO_LIST_QN)))
        collector_func = FunctionApplication(JAVA_STREAM_COLLECT_QN, [collectors_aggregate_func],
                                             method_target=container_func)
        return collector_func
    else:
        return container_func


def transform_container(term, container, method_target=None):
    if isinstance(container, ComprehensionContainer):
        # Transform sub container if necessary
        if isinstance(container.container, (Aggregate, Stream)):
            add_collector = False
            if add_collector:
                stream_target = visit_aggregate(container.container, add_collector)
                stream_func = FunctionApplication(JAVA_STREAM_QN, [], stream_target)
            else:
                stream_func = visit_aggregate(container.container, add_collector)
        elif isinstance(container.container, (FunctionApplication, MathVariable)):
            stream_func = container.container
        else:
            stream_func = FunctionApplication(JAVA_STREAM_QN, [], container.container)

        # Transform rest
        if container.rest:
            if isinstance(container.rest, ComprehensionCondition):
                sub_condition_func = transform_condition(term, container, container.rest, stream_func)
                return sub_condition_func
            if isinstance(container.rest, ComprehensionContainer):
                sub_container_func = transform_container(term, container.rest, method_target=None)
                args = [to_qn_lambda(var) for var in container.vars]
                flat_map_lambda = LambdaExpression(args, sub_container_func)
                flat_map_func = FunctionApplication(JAVA_STREAM_FLATMAP_QN, [flat_map_lambda],
                                                    method_target=method_target or stream_func)

                return flat_map_func
        else:
            args = [to_qn_lambda(var) for var in container.vars]
            return FunctionApplication(JAVA_STREAM_MAP_QN, [LambdaExpression(args, term)], method_target=stream_func)


def transform_condition(term, container, condition, method_target=None):
    args = [to_qn_lambda(var) for var in container.vars]
    condition_lambda = LambdaExpression(args, condition.condition)
    filter_func = FunctionApplication(JAVA_STREAM_FILTER_QN, [condition_lambda], method_target=method_target)
    if condition.rest:
        if isinstance(condition.rest, ComprehensionCondition):
            return transform_condition(term, container, condition.rest, filter_func)
        if isinstance(condition.rest, ComprehensionContainer):
            container_func = transform_container(term, condition.rest, method_target=None)
            if method_target:
                args = [to_qn_lambda(var) for var in container.vars]
                flat_map_lambda = LambdaExpression(args, container_func)
                flat_map_func = FunctionApplication(JAVA_STREAM_FLATMAP_QN, [flat_map_lambda],
                                                    method_target=filter_func)
                return flat_map_func
            else:
                return container_func
    else:
        return FunctionApplication(JAVA_STREAM_MAP_QN, [LambdaExpression(args, term)], method_target=filter_func)
