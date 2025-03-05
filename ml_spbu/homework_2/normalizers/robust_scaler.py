from ml_spbu.homework_2.annotations import Point
from ml_spbu.homework_2.exceptions import NotFittedError
from ml_spbu.homework_2.normalizers.abstract_scaler import AScaler


class RobustScaler(AScaler):
    def __init__(self):
        self._iqr = None
        self._q2 = None
        self._features_number = None

    def fit(self, X: list[Point]):
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

    def transform(self, X: list[Point]):
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

    def fit_transform(self, X: list[Point]):
        self.fit(X)
        return self.transform(X)
