from typing import Any


class SingletonMeta(type):
    """
    Metaclass implementing the Singleton design pattern.

    Only one instance of classes using this metaclass will be created.
    """

    _instances: dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
