from math import inf

from ml_spbu.homework_2.annotations import Point
from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.abstract_scaler import AScaler


class MinMaxScaler(AScaler):
    def __init__(self):
        self._data_min = None
        self._data_max = None
        self._features_number = None

    def fit(self, X: list[Point]):
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

    def transform(self, X: list[Point]):
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

    def fit_transform(self, X: list[Point]):
        self.fit(X)
        return self.transform(X)
