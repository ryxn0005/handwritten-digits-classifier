import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, figsize=(5, 4), dpi=100):
    """
    Plots a confusion matrix for the given true and predicted labels.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels of the data.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels by the model.
    - figsize: tuple, optional (default=(5, 4))
        Size of the figure (width, height) in inches.
    - dpi: int, optional (default=100)
        Resolution of the plot in dots per inch.

    Returns:
    - Displays a confusion matrix plot.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Initialize and display the confusion matrix
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmp.plot(ax=ax)

    # Customize plot aesthetics
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
