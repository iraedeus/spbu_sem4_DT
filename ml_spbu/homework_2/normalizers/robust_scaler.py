"""Module implementing Robust Scaling for feature normalization.

This module provides the RobustScaler class, which scales features using statistics
that are robust to outliers. It removes the median and scales the data according to
the Interquartile Range (IQR).
"""

from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.abstract_scaler import AScaler


class RobustScaler(AScaler):
    """Robust Scaler for feature scaling using IQR.

    This scaler removes the median and scales the data according to the IQR.
    The IQR is the range between the 1st quartile (25th percentile) and the
    3rd quartile (75th percentile). Centering and scaling happen independently
    on each feature by computing the relevant statistics on the samples in the
    training set. Median and IQR are then stored to be used for transform on
    later data using the `transform` method.

    Attributes:
        _iqr (list[float] | None): Interquartile Range for each feature, computed during fit.
        _q2 (list[float] | None): Median (2nd quartile) for each feature, computed during fit.
        _features_number (int | None): Number of features in the input data.
    """

    def __init__(self):
        """Initialize RobustScaler with default parameters.

        Sets initial values for IQR, median, and feature number to None.
        These attributes are computed during the `fit` method.
        """

        self._iqr = None
        self._q2 = None
        self._features_number = None

    def fit(self, X):
        """Compute the median and IQR for each feature in the training set.

        Calculates the median (Q2) and Interquartile Range (IQR) for each feature
        in the input dataset `X`. These values are stored and used later in the
        `transform` method to perform robust scaling.

        Args:
            X (list[Point]): The training data used to compute medians and IQRs.
                Each Point represents a data sample, and the elements of the tuple
                represent the features.
        """

        self._features_number = len(X[0])
        n = self._features_number

        iqr = []
        q2 = []

        for i in range(n):
            feature_values = [x[i] for x in X]
            feature_values.sort()
            m = len(feature_values)

            q1_rank = int(1 / 4 * (m - 1))
            q2_rank = int(1 / 2 * (m - 1))
            q3_rank = int(3 / 4 * (m - 1))

            iqr.append(feature_values[q3_rank] - feature_values[q1_rank])
            q2.append(feature_values[q2_rank])

        self._iqr = iqr
        self._q2 = q2

    def transform(self, X):
        """Scale features of X using the median and IQR computed during fit.

        Performs robust scaling on the input data `X` using the formula:
        `(x - median) / IQR`. If IQR is zero for a feature, the feature is not scaled
        and the value becomes 0.

        Args:
            X (list[Point]): The data to be scaled.

        Returns:
            list[Point]: The robustly scaled data.

        Raises:
            NotFittedError: If the `fit` method has not been called before `transform`.
        """

        if not self._iqr or not self._q2:
            raise NotFittedError()

        X_scaled = []
        n = self._features_number

        iqr = self._iqr
        q2 = self._q2

        for x in X:
            x_scaled = []
            for i in range(n):
                if iqr[i] == 0:
                    x_scaled.append(0)
                else:
                    x_scaled.append((x[i] - q2[i]) / iqr[i])

            X_scaled.append(tuple(x_scaled))

        return X_scaled

    def fit_transform(self, X):
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
