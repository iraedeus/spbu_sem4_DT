def f1_score(y_pred: list[int], y_true: list[int]) -> float:
    if len(y_pred) != len(y_true):
        raise AttributeError("The lengths of arrays y_pred and y_true must be the same.")

    true_positives = sum(pred == true == 1 for pred, true in zip(y_pred, y_true))
    false_positives = sum((pred == 1) and (pred != true) for pred, true in zip(y_pred, y_true))
    false_negatives = sum((pred == 0) and (pred != true) for pred, true in zip(y_pred, y_true))

    if true_positives == 0:
        return 0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return 2 * ((precision * recall) / (precision + recall))
