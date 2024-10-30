def calculate_multiclass_metrics(y_true, y_pred, num_classes=10):
    metrics = {}

    for class_label in range(num_classes):
        TP = FP = TN = FN = 0
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                if pred == class_label:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred == class_label:
                    FP += 1
                elif true == class_label:
                    FN += 1

        # Calculate metrics for each class
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Initialize dictionary for the current class_label
        metrics[class_label] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    return metrics
