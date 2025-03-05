"""Module implementing the Min-Max feature scaling.

This module provides the MinMaxScaler class, which scales features by
transforming each feature between zero and one.
This scaling is achieved by subtracting the minimum value of each feature and
then dividing by the range (max - min).
"""

from math import inf

from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.abstract_scaler import AScaler


class MinMaxScaler(AScaler):
    """Min-Max Scaler for feature scaling.

    This scaler transforms features by scaling each feature to a given range,
    by default between zero and one. The transformation is given by::

        X_scaled[i] = (X[i] - X[axis = i].min) / (X[axis = i].max - X[axis = i].min)

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Attributes:
        _data_min (list[float] | None): Minimum value for each feature in the training set.
        _data_max (list[float] | None): Maximum value for each feature in the training set.
        _features_number (int | None): Number of features in the input data.
    """

    def __init__(self):
        """Initialize MinMaxScaler with default parameters.

        There are no parameters to set upon initialization.
        The scaler is initialized with None values for data_min, data_max,
        and features_number, which are computed during the `fit` method call.
        """

        self._data_min = None
        self._data_max = None
        self._features_number = None

    def fit(self, X):
        """Compute the minimum and maximum values for each feature in the dataset.

        This method calculates the minimum and maximum values for each feature
        across the input dataset `X`. These values are stored and used later
        in the `transform` method to scale the data.

        Args:
            X (list[Point]): The input data used to compute the min and max values.
                Each Point represents a data sample, and the elements of the tuple
                represent the features.
        """

        self._features_number = len(X[0])
        n = self._features_number

        data_min = [inf for _ in range(n)]
        data_max = [-inf for _ in range(n)]

        for x in X:
            for i, feature in enumerate(x):
                data_min[i] = min(feature, data_min[i])
                data_max[i] = max(feature, data_max[i])

        self._data_min = data_min
        self._data_max = data_max

    def transform(self, X):
        """Scale features of X according to the min and max values computed during fit.

        For each feature, this method scales the input data `X` using the
        formula: `(x - min) / (max - min)`, where min and max are the
        minimum and maximum values computed for that feature during the `fit` method.
        If `max` is equal to `min` for a feature, it is set to 0 for that feature.

        Args:
            X (list[Point]): The input data to be scaled.

        Returns:
            list[Point]: The scaled data.

        Raises:
            NotFittedError: If the `fit` method has not been called before `transform`.
        """

        if not self._data_max or not self._data_min:
            raise NotFittedError()

        X_scaled = []
        n = self._features_number

        data_min = self._data_min
        data_max = self._data_max

        for x in X:
            x_scaled = []
            for i in range(n):
                if data_max[i] == data_min[i]:
                    x_scaled.append(0)
                else:
                    x_scaled.append((x[i] - data_min[i]) / (data_max[i] - data_min[i]))

            X_scaled.append(tuple(x_scaled))

        return X_scaled

    def fit_transform(self, X):
        """Fit to data, then transform it.

        Fits the scaler to `X` and then transforms `X`.
        It's equivalent to calling `fit(X)` followed by `transform(X)`.

        Args:
            X (list[Point]): The input data.

        Returns:
            list[Point]: The transformed data.
        """

        self.fit(X)
        return self.transform(X)
