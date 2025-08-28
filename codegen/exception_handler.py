from functools import wraps
from inspect import ismethod
from operator import methodcaller, attrgetter


def is_method(obj, name):
    return hasattr(obj, name) and ismethod(getattr(obj, name))


def is_attribute(obj, name):
    return hasattr(obj, name)


def silent_exception(method):
    @wraps(method)
    def silent(self, *args, **kw):
        try:
            return method(self, *args, **kw)
        except Exception as e:
            line = 'no information'
            if args[0] is not None :
                obj = args[0]
                attr = 'starts_on_line'
                if is_attribute(obj, attr):
                    line = attrgetter(attr)(obj)
            print(f'Exception [{self.__class__.__name__}.{method.__name__}], msg = [{e}], input line [{line}]')

    return silent
