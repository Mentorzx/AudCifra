from typing import Any, Protocol


class Observer(Protocol):
    """
    Protocol for observers.

    :return: None.
    """

    def update(self, event: str, data: Any) -> None: ...


class Subject:
    """
    Subject class implementing the Observer design pattern.

    :return: None.
    """

    def __init__(self):
        self._observers: list[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self, event: str, data: Any) -> None:
        for observer in self._observers:
            observer.update(event, data)
