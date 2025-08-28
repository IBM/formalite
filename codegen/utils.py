import functools
from fnmatch import fnmatchcase
from functools import wraps
from inspect import isclass, getmembers, ismethod, isfunction, signature
from operator import methodcaller
from typing import Iterator, Optional, Callable
from weakref import WeakValueDictionary

import inflection
from itertools import count


def is_method(obj, name):
    return hasattr(obj, name) and ismethod(getattr(obj, name))


def decorate_method(*decorators_and_method_patterns, supermetaclass=type):
    """
    Create a metaclass that will apply `decorator` to any method matching `method_pattern`

    Matching methods will be decorated in all inheriting subclasses.

    :param decorator: a method decorator
    :param method_pattern: an `fnmatch` pattern
    :param supermetaclass: superclass of generated metaclass (e.g., `type` or `ABCMeta`)
    :return: new metaclass
    """
    decorators_and_method_patterns = list(reversed(decorators_and_method_patterns))

    class DecoratingMetaClass(supermetaclass):
        def __init__(cls, class_name, supers, class_dict):
            for method_name, body in class_dict.items():
                for decorator, method_pattern in decorators_and_method_patterns:
                    if fnmatchcase(method_name, method_pattern):
                        body = decorator(body)
                        setattr(cls, method_name, body)
            super().__init__(class_name, supers, class_dict)

    return DecoratingMetaClass


# Better use Acceptor below, which is not a metaclass
class AcceptingMetaClass:
    """
    Add accept method according to the visitor pattern to all subclasses.
    """

    def __init__(cls, class_name, supers, class_dict):
        visitor_method = f'visit_{inflection.underscore(cls.__name__)}'

        def accept(self, obj):
            return methodcaller(visitor_method, self)(obj)

        setattr(cls, visitor_method, accept)


class Acceptor:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        visitor_method = f'visit_{inflection.underscore(cls.__name__)}'

        def accept(self, visitor, *args, **kwargs):
            return getattr(visitor, visitor_method)(self, *args, **kwargs)
            # return methodcaller(visitor_method, self)(visitor)

        setattr(cls, 'accept', accept)
        # print(f'Added accept method to {cls.__name__}, calling {visitor_method}')


def collect_subclasses(superclass, result=None):
    if result is None:
        result = []
    for cls in superclass.__subclasses__():
        result.append(cls)
        collect_subclasses(cls, result)
    return result


def disown(*vars):
    """
    Annotate the __init__ method by the variables that are not "children" and shouldn't be recursively visited
    :param vars: sequence of parameter names of the __init__method that should be ignored
    """

    def decorator(init_method):
        if isinstance(init_method, type):
            getattr(init_method, '__init__')._disowned_vars = vars
        else:
            init_method._disowned_vars = vars
        return init_method

    return decorator


def make_children_visitor(object_class, child_names, add_call_to=None, collect_results=True):
    """
    Create a function that visits the children of an object of object_class  The children are identified by the
    parameters of the __init__ method, but can be excluded by the @disown decorator.

    N.B. Children are NOT identified by assignments to attributes of self in the constructor, but by the names
    of the constructor parameters.  This assumes that the attributes names are identical to the parameter names!

    :param object_class: class of object being visited
    :param child_names: names of children attributes to be visited
    :param add_call_to: the name of a method that should be called automatically for every visited node
    :param collect_results: True iff visitor methods for internal nodes should return the list of children values
    :return: visitor function
    """

    if collect_results:
        if add_call_to:
            def visit_c(self, x: object_class, *args, **kwargs):
                results = []
                for child in child_names:
                    val = getattr(x, child)
                    if val is not None:
                        if isinstance(val, (list, tuple)):
                            for e in val:
                                results.append(self.visit(e, *args, **kwargs))
                        else:
                            results.append(self.visit(val, *args, **kwargs))
                getattr(self, add_call_to)(x, *args, **kwargs)
                return results
        else:
            def visit_c(self, x: object_class, *args, **kwargs):
                results = []
                for child in child_names:
                    val = getattr(x, child)
                    if val is not None:
                        if isinstance(val, (list, tuple)):
                            for e in val:
                                results.append(self.visit(e, *args, **kwargs))
                        else:
                            results.append(self.visit(val, *args, **kwargs))
                return results
    else:
        # don't collect results
        if add_call_to:
            def visit_c(self, x: object_class, *args, **kwargs):
                for child in child_names:
                    val = getattr(x, child)
                    if val is not None:
                        if isinstance(val, (list, tuple)):
                            for e in val:
                                self.visit(e, *args, **kwargs)
                        else:
                            self.visit(val)
                getattr(self, add_call_to)(x, *args, **kwargs)
        else:
            def visit_c(self, x: object_class, *args, **kwargs):
                for child in child_names:
                    val = getattr(x, child)
                    if val is not None:
                        if isinstance(val, (list, tuple)):
                            for e in val:
                                self.visit(e, *args, **kwargs)
                        else:
                            self.visit(val, *args, **kwargs)

    return visit_c


def visitor_for(visited_class, no_recursion=False, add_call_to=None, collect_results=True):
    """
    Create visit method in the decorated class, each corresponding to the subclasses of the given class

    The default behavior is to visit all children of the node.  The children are identified by the
    parameters of the __init__ method, but can be excluded by the @disown decorator.

    N.B. Children are NOT identified by assignments to attributes of self in the constructor, but by the names
    of the constructor parameters.  This assumes that the attributes names are identical to the parameter names!

    :param visited_class: a class whose subclasses are to be visited
    :param no_recursion: don't recurse to children
    :param add_call_to: the name of a method that should be called automatically for every visited node
    :param collect_results: True iff visitor methods for internal nodes should return the list of children values
    :return: the decorated class, with added visit* methods
    """

    def add_methods(visitor_class):
        if not isclass(visited_class):
            raise Exception('@visitor_for requires a class parameter')
        if not isclass(visitor_class):
            raise Exception('@visitor_for can only decorate classes')
        if add_call_to and not hasattr(visitor_class, add_call_to):
            raise Exception(f'Method {add_call_to} does not exist in class {visitor_class.__name__}')

        def visit(self, obj, *args, **kwargs):
            return obj.accept(self, *args, **kwargs)

        setattr(visitor_class, 'visit', visit)

        subclasses = collect_subclasses(visited_class)
        for c in subclasses:
            pars = ()
            # TODO: replace with:     init_method = getattr(cls, '__init__', None)
            if (not no_recursion and
                    (init_method := next((m[1] for m in getmembers(c, isfunction) if m[0] == '__init__'), None))
                    is not None):
                pars = tuple(k for k in signature(init_method).parameters)[1:]
                if (ignored := getattr(init_method, '_disowned_vars', None)) is not None:
                    pars = tuple(p for p in pars if p not in ignored)
            if pars:
                # N.B. must have a binding of pars that doesn't change!
                visit_c = make_children_visitor(c, pars, add_call_to, collect_results=collect_results)
            else:
                if add_call_to:
                    def visit_c(self, x: c, *args, **kwargs):
                        getattr(self, add_call_to)(x, *args, **kwargs)
                else:
                    def visit_c(self, x: c, *args, **kwargs):
                        pass

            m_name = f'visit_{inflection.underscore(c.__name__)}'
            visit_c.__name__ = m_name
            setattr(visitor_class, m_name, visit_c)
            # print(f'Added visitor method {m_name}({", ".join(pars)})')
        return visitor_class

    return add_methods


def uniqueize_name(proposed_name, existing_names):
    if proposed_name in existing_names:
        prefix = proposed_name + '_'
        # change to unique name
        clashes = [name[len(prefix):] for name in existing_names if name.startswith(prefix)]
        suffix = max((int(s) for s in clashes if s.isdigit()), default=0) + 1
        proposed_name = prefix + str(suffix)
    return proposed_name


class Intern:
    """
    A class of interning decorators.

    Each instance of this class can be used as a decorator of a class. This will modify or add ``__new__`` and
    ``__init__`` methods to the decorated class so that instances of the class will be reused instead of creating
    multiple objects with the same value. (See :class:`str:intern` for a specific example of this behavior.)

    Objects are considered to be the same if they are created with the same parameters in the constructor call. These
    parameters are used as a key to an internal table that keeps track of already-created class objects. By default,
    the key consists of the values of the parameters, which must therefore be hashable.

    If some constructor parameters may be unhashable, a key function must be provided (either as a ``key``
    parameter to the specific ``intern`` call, or as a general ``default_key`` for the ``Intern`` class
    constructor). The key function will be used to compute the interning key; it will be given the same
    parameters given to the constructor.

    Note that a constructor call that provides a keyword parameter with its default value will create an object
    different from one created with a call that does not provide this keyword parameter. This behavior can be
    overridden by a custom key.

    For a detailed explanation, see the blog post
    https://the-dusty-deck.blogspot.com/2022/12/metaprogramming-in-python-2-interning.html
    """

    resettable_instances = []

    def __new__(cls, *args, **kwargs):
        result = super().__new__(cls)
        if kwargs.get('reset_intern_memory', 'default') is not None:
            cls.resettable_instances.append(result)
        return result

    @classmethod
    def reset_all(cls):
        for interner in cls.resettable_instances:
            for interned_class in interner.interned_classes:
                interned_class.reset_intern_memory()

    def __init__(self, default_key: Optional[Callable[[type], Callable]] = None, weak_dict=True):
        """
        Create an interning decorator.

        :param default_key: an optional function that returns an interning key given the decorated class
        :param weak_dict indicates whether the memory should be a WeakValueDictionary, which will forget
        values that aren't cached elsewhere; however, if the class has a field named ``_Intern__strong__dict_``,
        a WeakValueDictionary will not be used for that class regardless of the value of this parameter.
        """
        self.weak_dict = weak_dict
        self.default_key = default_key
        self.interned_classes = []

    def __call__(self, cls: Optional[type] = None, *,
                 key: Callable = None,
                 eq_method=None,
                 hash_method=None,
                 native_key=False,
                 reset_method: Optional[str] = 'reset_intern_memory'):
        """
        Decorator that interns elements of the class based on the parameters of the __init__ method.

        The parameters of the __init__ method of the class must all be hashable.

        :param cls: class whose elements are to be interned
        :param key: an optional key function
        :param reset_method: the name of a method to be added to the class for resetting the interning table;
        None to prevent the creation of this method
        :return: same class, with modified or added __new__ and  __init__ methods
        """
        if cls is None:
            return functools.partial(self, key=key, eq_method=eq_method, hash_method=hash_method,
                                     native_key=native_key, reset_method=reset_method)
        if key is None and not native_key and hasattr(cls, '_interning_key') and callable(cls._interning_key):
            key = cls._interning_key

        if key is None and self.default_key is not None:
            key = self.default_key(cls)

        memory = WeakValueDictionary() if self.weak_dict and not hasattr(cls, '_Intern__strong__dict_') else {}
        cls._intern_memory = memory  # DEBUG

        old_new = getattr(cls, '__new__')
        if old_new is object.__new__:
            def __new__(cls, *args, **kwargs):
                element_key = (args, tuple(sorted(kwargs.items()))) if key is None else key(*args, **kwargs)
                try:
                    result = memory[element_key]
                    setattr(result, '*already-initialized*', True)
                    return result
                except KeyError:
                    result = object.__new__(cls)
                    memory[element_key] = result
                    return result
        else:
            def __new__(cls, *args, **kwargs):
                element_key = (args, tuple(sorted(kwargs.items()))) if key is None else key(*args, **kwargs)
                try:
                    result = memory[element_key]
                    setattr(result, '*already-initialized*', True)
                    return result
                except KeyError:
                    result = old_new(cls, *args, **kwargs)
                    memory[element_key] = result
                    return result

        setattr(cls, '__new__', __new__)

        if eq_method:
            def __eq__(self, other):
                return self is other

            setattr(cls, '__eq__' if eq_method is True else eq_method, __eq__)

        if hash_method:
            def __hash__(self):
                return id(self)

            setattr(cls, '__hash__' if hash_method is True else hash_method, __hash__)

        init = getattr(cls, '__init__', None)
        if init is None:
            def __init__(self, *args, **kwargs):
                if not (hasattr(self, '*already-initialized*') and getattr(self, '*already-initialized*')):
                    super().__init__(self, *args, **kwargs)
        else:
            @wraps(init)
            def __init__(self, *args, **kwargs):
                if not (hasattr(self, '*already-initialized*') and getattr(self, '*already-initialized*')):
                    init(self, *args, **kwargs)

        setattr(cls, '__init__', __init__)

        if reset_method is not None:
            def reset_intern_method(weak_dict=None):
                if weak_dict is not None:
                    self.weak_dict = weak_dict
                nonlocal memory
                memory = WeakValueDictionary() if self.weak_dict and not hasattr(cls, '_Intern__strong__dict_') else {}
                # cls._intern_memory = memory  # DEBUG

            setattr(cls, reset_method, reset_intern_method)
            self.interned_classes.append(cls)
        return cls


intern = Intern()


def spread(indexes, values, pad=None) -> Iterator:
    """
    Given a strictly increasing series of indexes and a corresponding series of values, return an infinite series that
    contains at each index the corresponding value, and the ``pad`` value in other places.
    """
    indexes = iter(indexes)
    values = iter(values)
    next_index = next(indexes, None)
    for current_index in count():
        if current_index == next_index:
            next_index = next(indexes, None)
            yield next(values)
        else:
            yield pad


def selective_combine(seq1, seq2, yield_value=None):
    """
    Return a series of the elements of ``seq1`` that are different from ``yield_value``; skipped values are replaced
    by the corresponding values from ``seq2``
    """
    return (s1 if s1 != yield_value else s2
            for s1, s2 in zip(seq1, seq2))
