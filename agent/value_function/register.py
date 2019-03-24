registry = {}


def register(cls):
    registry[cls.__name__] = cls
    return cls