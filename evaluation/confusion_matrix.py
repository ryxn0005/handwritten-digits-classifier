import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, figsize=(5, 4), dpi=100):
    """
    Plots a confusion matrix for the given true and predicted labels.

    @Usage:
        Generates a confusion matrix to evaluate the performance of a classification model
        by comparing true labels with predicted labels.

    @Parameters:
    y_true : array-like of shape (n_samples,)
        True labels of the data.
    y_pred : array-like of shape (n_samples,)
        Predicted labels by the model.
    figsize : tuple, optional, default=(5, 4)
        Size of the figure (width, height) in inches.
    dpi : int, optional, default=100
        Resolution of the plot in dots per inch.

    @Returns:
    None
        Displays the confusion matrix plot.
    """
    # Compute the confusion matrix from true and predicted labels
    cm = confusion_matrix(y_true, y_pred)

    # Set up the plot with specified figure size and resolution
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Initialize ConfusionMatrixDisplay with computed matrix and plot it on the axes
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmp.plot(ax=ax)

    # Customize plot aesthetics
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
