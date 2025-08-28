# Utilities that are used in optimistic_client code as well as optimistic_client runtime

class KeepMembersMixin:
    def __new__(cls, *args, **kwargs):
        newobj = object.__new__(cls)
        newobj.__init__(*args, **kwargs)
        members = getattr(cls, '**members**', None)
        if members is None:
            members = set()  # WeakSet()
            setattr(cls, '**members**', members)
        members.add(newobj)
        return newobj

    @classmethod
    def members(cls):
        return getattr(cls, '**members**', ())
