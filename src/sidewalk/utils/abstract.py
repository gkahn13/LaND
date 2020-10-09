import sys
assert sys.version_info.major == 3
if sys.version_info.minor >= 6:
    import abc as _abc
    from abc import abstractmethod
    import overrides as _overrides
    from overrides import final, overrides

    class BaseClass(_abc.ABC, _overrides.EnforceOverrides):
        pass
else:
    print('!!!!!!!!!! ABSTRACT NOT BEING USED')
    def null_decorator(obj):
        return obj
    abstractmethod = final = overrides = null_decorator

    class BaseClass(object):
        pass

