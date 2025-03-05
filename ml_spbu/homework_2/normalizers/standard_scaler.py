"""Module implementing Standard Scaling for feature normalization.

This module provides the StandardScaler class, which standardizes features by
removing the mean and scaling to unit variance. Standardization is a common
preprocessing step in machine learning.
"""

from math import sqrt

from ml_spbu.homework_2.annotations import Point
from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.abstract_scaler import AScaler


class StandardScaler(AScaler):
    """Standard Scaler for feature standardization.

    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples and `s` is the standard
    deviation of the training samples.

    Attributes:
        _means (list[float] | None): Mean value for each feature in the training set.
        _deviations (list[float] | None): Standard deviation for each feature in the training set.
        _features_number (int | None): Number of features in the input data.
    """

    def __init__(self):
        """Initialize StandardScaler with default parameters.

        Sets initial values for means, deviations, and feature number to None.
        These attributes are computed during the `fit` method.
        """

        self._means = None
        self._deviations = None
        self._features_number = None

    def fit(self, X: list[Point]):
        """Compute the mean and standard deviation for each feature in the training set.

        Calculates the mean and standard deviation for each feature across the
        input dataset `X`. These statistics are stored and used later in the
        `transform` method to standardize the data.

        Args:
            X (list[Point]): The training data used to compute means and standard deviations.
                Each Point represents a data sample, and the elements of the tuple
                represent the features.
        """

        self._features_number = len(X[0])
        n = self._features_number
        m = len(X)

        means = []
        deviations = []

        for i in range(n):
            mean_i = sum(x[i] for x in X) / m
            deviation_i = sqrt(sum((x[i] - mean_i) ** 2 for x in X) / m)

            means.append(mean_i)
            deviations.append(deviation_i)

        self._means = means
        self._deviations = deviations

    def transform(self, X: list[Point]):
        """Perform standardization by centering and scaling.

        For each feature, subtracts the mean and divides by the standard deviation,
        using the statistics computed during the `fit` method. If standard deviation
        is zero for a feature, the feature is set to zero.

        Args:
            X (list[Point]): The data to be standardized.

        Returns:
            list[Point]: The standardized data.

        Raises:
            NotFittedError: If the `fit` method has not been called before `transform`.
        """

        if not self._means or not self._deviations:
            raise NotFittedError()

        X_scaled = []
        n = self._features_number

        means = self._means
        deviations = self._deviations

        for x in X:
            x_scaled = []
            for i in range(n):
                if deviations[i] == 0:
                    x_scaled.append(0)
                else:
                    x_scaled.append((x[i] - means[i]) / deviations[i])
            X_scaled.append(tuple(x_scaled))

        return X_scaled

    def fit_transform(self, X: list[Point]):
        """Fit to data, then transform it.

        Fits the scaler to `X` and then transforms `X`.
        Equivalent to calling `fit(X)` followed by `transform(X)`.

        Args:
            X (list[Point]): The input data.

        Returns:
            list[Point]: The transformed data.
        """

        self.fit(X)
        return self.transform(X)
