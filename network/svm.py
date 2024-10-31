# from sklearn.svm import SVC
from cuml.svm import SVC
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
import cupy as cp
from sklearn.metrics import accuracy_score
import json


class SVMImageClassifier(SVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.abbreviation = "svm"

    def _flatten_images(self, X):
        """
        Flattens each image from 2D to 1D if not already flattened.
        Expects input X of shape (n_samples, height, width, channels).
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
        Fit the model on the entire dataset and track training and validation history.
        """
        X_flat = self._flatten_images(X)

        # Convert to CuPy arrays for GPU processing
        X_flat = cp.asarray(X_flat)
        y = cp.asarray(y)

        # Split the data into training and validation sets if X_val and y_val are not provided
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

        # Calculate training accuracy and loss
        y_train_pred = super().predict(X_train)
        train_accuracy = accuracy_score(cp.asnumpy(y_train), cp.asnumpy(y_train_pred))

        # Note: cuML's SVC does not provide `predict_proba` by default (since SVMs do not inherently output probabilities)
        # If you require probabilities, consider using a method like Platt scaling after fitting the model
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

        # Save history log
        self._save_log(log)

        return log

    def predict(self, X):
        """
        Override the predict method to accept 2D image input and flatten it.
        """
        X_flat = self._flatten_images(X)
        return super().predict(X_flat)

    def save(self, save_dir: str = "./model/"):
        # Define the path based on model abbreviation
        model_path = os.path.join(save_dir, self.abbreviation)
        os.makedirs(model_path, exist_ok=True)  # Ensure directory exists

        # Save the data using the model abbreviation for the filename
        processed_data_path = os.path.join(model_path, f"{self.abbreviation}.pkl")

        # Save the model
        with open(processed_data_path, "wb") as pickle_out:
            pickle.dump(self, pickle_out)
        print(f"Model saved to '{processed_data_path}'")

    def _save_log(self, log):
        """
        Save the training history to a log file in JSON format.
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
