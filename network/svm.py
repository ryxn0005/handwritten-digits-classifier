from cuml.svm import SVC
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
import cupy as cp
from sklearn.metrics import accuracy_score
import json


class SVMImageClassifier(SVC):
    """
    SVMImageClassifier is a subclass of cuML's SVC designed to classify images
    with support for GPU acceleration and custom training and logging functionality.

    @Parameters:
    kwargs : dict
        Additional arguments to initialize the SVC model.

    @Attributes:
    abbreviation : str
        Abbreviation used to name saved model files and logs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.abbreviation = "svm"

    def _flatten_images(self, X):
        """
        Flattens images from 2D to 1D for compatibility with the SVM model.

        @Usage:
            Reshapes each image in the batch from (height, width, channels) to a single-dimensional array.
            Assumes input X has shape (n_samples, height, width, channels) or (n_samples, flattened_dim).

        @Parameters:
        X : np.ndarray
            Batch of images, either in original or already flattened form.

        @Returns:
        np.ndarray : Flattened images with shape (n_samples, flattened_dim).
        """
        if len(X.shape) == 2:  # Already flattened
            return X
        elif len(X.shape) == 4:  # Needs flattening
            n_samples, height, width, in_chans = X.shape
            return X.reshape(n_samples, height * width * in_chans)
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")

    def train(self, X, y, validation_split=0.2, X_val=None, y_val=None):
        """
        Trains the SVM classifier on the provided dataset, calculates accuracy, and logs training history.

        @Usage:
            Splits the data into training and validation sets, fits the model, and evaluates performance.

        @Parameters:
        X : np.ndarray
            Input training data.
        y : np.ndarray
            Labels corresponding to the input data.
        validation_split : float, optional, default=0.2
            Proportion of the dataset to use for validation if validation data is not provided.
        X_val : np.ndarray, optional
            Validation data features.
        y_val : np.ndarray, optional
            Validation data labels.

        @Returns:
        dict : Training and validation metrics, including accuracy.
        """
        X_flat = self._flatten_images(X)

        # Convert to CuPy arrays for GPU processing
        X_flat = cp.asarray(X_flat)
        y = cp.asarray(y)

        # Split the data if no validation data is provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y, test_size=validation_split, random_state=42
            )
        else:
            X_val_flat = self._flatten_images(X_val)
            X_val = cp.asarray(X_val_flat)
            X_train, y_train, X_val, y_val = X_flat, y, X_val, y_val

        # Fit the model on the training set
        super().fit(X_train, y_train)

        # Calculate training accuracy
        y_train_pred = super().predict(X_train)
        train_accuracy = accuracy_score(cp.asnumpy(y_train), cp.asnumpy(y_train_pred))

        # Note: cuML's SVC does not provide `predict_proba` by default
        train_loss = None

        # Calculate validation accuracy
        y_val_pred = super().predict(X_val)
        val_accuracy = accuracy_score(cp.asnumpy(y_val), cp.asnumpy(y_val_pred))
        val_loss = None

        log = {
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
        }

        print(
            f"Training complete - train_accuracy: {train_accuracy:.4f} "
            f"- val_accuracy: {val_accuracy:.4f}"
        )

        # Save training log
        self._save_log(log)

        return log

    def predict(self, X):
        """
        Predict class labels for the input data, flattening images if necessary.

        @Usage:
            Accepts 2D image input, flattens it if needed, and returns class predictions.

        @Parameters:
        X : np.ndarray
            Input data to predict class labels.

        @Returns:
        np.ndarray : Predicted class labels.
        """
        X_flat = self._flatten_images(X)
        return super().predict(X_flat)

    def save(self, save_dir: str = "./model/"):
        """
        Save the trained model to a specified directory in pickle format.

        @Usage:
            Serializes the model instance as a .pkl file for future use.

        @Parameters:
        save_dir : str, optional, default="./model/"
            Directory where the model will be saved.

        @Returns:
        None
        """
        # Define the path based on model abbreviation
        model_path = os.path.join(save_dir, self.abbreviation)
        os.makedirs(model_path, exist_ok=True)  # Ensure directory exists

        # Save the model using the abbreviation for the filename
        processed_data_path = os.path.join(model_path, f"{self.abbreviation}.pkl")
        with open(processed_data_path, "wb") as pickle_out:
            pickle.dump(self, pickle_out)
        print(f"Model saved to '{processed_data_path}'")

    def _save_log(self, log):
        """
        Save the training history to a log file in JSON format with a timestamp.

        @Usage:
            Logs metrics such as training accuracy, loss, validation accuracy, and loss.

        @Parameters:
        log : dict
            Dictionary containing the training metrics and losses.

        @Returns:
        None
        """
        log_dir = f"./logs/{self.abbreviation}/"
        os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"log-{timestamp}.json")

        log_serializable = {
            k: float(v) if v is not None else None for k, v in log.items()
        }

        # Save the history as a JSON file
        with open(log_path, "w") as log_file:
            json.dump(log_serializable, log_file, indent=4)
        print(f"Training history saved to '{log_path}'")
