from abc import ABC, abstractmethod

from ml_spbu.homework_2.annotations import Point


class AScaler(ABC):
    @abstractmethod
    def fit(self, X: list[Point]): ...

    @abstractmethod
    def transform(self, X: list[Point]): ...

    @abstractmethod
    def fit_transform(self, X: list[Point]): ...
