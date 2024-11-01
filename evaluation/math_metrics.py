def calculate_multiclass_metrics(y_true, y_pred, num_classes=10):
    """
    Calculates accuracy, precision, recall, and F1-score for each class in a multiclass classification setting.

    @Usage:
        Computes evaluation metrics for each class individually, providing insights into per-class performance.
        Useful for assessing how well the model performs across all classes.

    @Parameters:
    y_true : array-like of shape (n_samples,)
        True labels of the data.
    y_pred : array-like of shape (n_samples,)
        Predicted labels by the model.
    num_classes : int, optional, default=10
        Number of unique classes in the dataset.

    @Returns:
    dict : A dictionary with each class as a key and a dictionary of its metrics as values.
        Each class dictionary contains 'accuracy', 'precision', 'recall', and 'f1_score'.
    """
    metrics = {}

    # Iterate over each class to calculate metrics independently
    for class_label in range(num_classes):
        TP = FP = TN = FN = (
            0  # Initialize counts for true positives, false positives, true negatives, false negatives
        )
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                if pred == class_label:
                    TP += 1  # True Positive
                else:
                    TN += 1  # True Negative
            else:
                if pred == class_label:
                    FP += 1  # False Positive
                elif true == class_label:
                    FN += 1  # False Negative

        # Calculate metrics for each class
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Initialize dictionary for the current class_label with calculated metrics
        metrics[class_label] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    return metrics
