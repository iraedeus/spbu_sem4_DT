"""Module defining an abstract base class for data scalers.

This module introduces the AScaler abstract base class, which serves as
an interface for implementing various data scaling techniques. Concrete
scaler classes should inherit from AScaler and implement its abstract methods.
"""

from abc import ABC, abstractmethod

from ml_spbu.homework_2.annotations import Point


class AScaler(ABC):
    """Abstract base class for data scalers.

    AScaler defines the common interface for all scaler classes in this library.
    Concrete scaler classes must inherit from AScaler and implement the
    `fit`, `transform`, and `fit_transform` methods to provide specific
    data scaling functionalities.

    This class is not intended to be instantiated directly.
    """

    @abstractmethod
    def fit(self, X: list[Point]):
        """Abstract method to fit the scaler to the input data.

        This method should compute the parameters necessary for scaling
        the data, such as mean and standard deviation in StandardScaler,
        or minimum and maximum values in MinMaxScaler.

        Args:
            X (list[Point]): The input data to fit the scaler on.
        """

    @abstractmethod
    def transform(self, X: list[Point]) -> list[Point]:
        """Abstract method to perform scaling on the input data.

        This method should apply the scaling transformation to the input data X,
        using the parameters learned in the `fit` method.

        Args:
            X (list[Point]): The input data to be transformed.

        Returns:
            list[Point]: The transformed data.
        """

    @abstractmethod
    def fit_transform(self, X: list[Point]) -> list[Point]:
        """Abstract method to fit the scaler and then transform the data.

        This method provides a shorthand for calling `fit` followed by `transform`
        on the same input data.

        Args:
            X (list[Point]): The input data to fit and transform.

        Returns:
            list[Point]: The transformed data after fitting.
        """
