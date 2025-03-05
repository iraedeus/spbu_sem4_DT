def accuracy_score(y_pred: list[int], y_true: list[int]) -> float:
    if len(y_pred) != len(y_true):
        raise AttributeError("The lengths of arrays y_pred and y_true must be the same.")

    if len(y_pred) == len(y_true) == 0:
        return 1.0

    correct_answers = sum(pred == true for pred, true in zip(y_pred, y_true))
    return correct_answers / len(y_pred)
