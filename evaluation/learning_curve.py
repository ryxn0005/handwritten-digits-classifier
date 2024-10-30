import matplotlib.pyplot as plt


def plot_learning_curve(history):
    """
    Plots the learning curve from the training history.

    Parameters:
    - history: Dictionary containing the training history with keys "batch", "samples_seen", and "accuracy".
    """
    # Extract data for plotting
    samples_seen = history["samples_seen"]
    accuracy = history["accuracy"]

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        samples_seen,
        accuracy,
        marker="o",
        linestyle="-",
        color="b",
        label="Training Accuracy",
    )
    plt.xlabel("Number of Samples Seen")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.legend()
    plt.show()
