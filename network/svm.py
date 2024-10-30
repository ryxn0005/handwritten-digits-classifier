from sklearn import svm
import numpy as np
import os
import pickle
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


class SVMImageClassifier(svm.SVC):
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
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

    def train(self, X, y, batch_size=100, epochs=10, validation_split=0.2):
        """
        Train the SVM model on flattened images with approximate batching and epochs.
        Logs history to a .pkl file with train/validation accuracy and loss per epoch.
        """
        X_flat = self._flatten_images(X)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, y, test_size=validation_split, random_state=42
        )
        n_samples = X_train.shape[0]

        # Initialize history dictionary to record metrics
        history = {
            "epoch": [],
            "train_accuracy": [],
            "train_loss": [],
            "val_accuracy": [],
            "val_loss": [],
        }

        for epoch in range(epochs):
            # Shuffle the training data at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Track support vectors and labels
            support_vectors = None
            support_labels = None

            for start in tqdm(
                range(0, n_samples, batch_size),
                desc=f"SVM | Epoch {epoch + 1}/{epochs}",
                unit="batch",
            ):
                end = min(start + batch_size, n_samples)
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                if start == 0:
                    # Initial fit on the first batch of the epoch
                    super().fit(X_batch, y_batch)
                    support_vectors = self.support_
                    support_labels = y_train[self.support_]
                else:
                    # Combine support vectors and fit with the new batch
                    X_combined = np.vstack([X_train[support_vectors], X_batch])
                    y_combined = np.hstack([support_labels, y_batch])

                    # Fit again with combined support vectors and new batch
                    super().fit(X_combined, y_combined)
                    # Update support vectors with the latest ones
                    support_vectors = self.support_
                    support_labels = y_combined[self.support_]

            # Calculate training accuracy and loss for the current epoch
            train_accuracy = super().score(X_train, y_train)
            train_probabilities = super().predict_proba(X_train)
            train_loss = log_loss(y_train, train_probabilities)
            # Calculate validation accuracy and loss for the current epoch
            val_accuracy = super().score(X_val, y_val)
            val_probabilities = super().predict_proba(X_val)
            val_loss = log_loss(y_val, val_probabilities)

            # Record history
            history["epoch"].append(epoch + 1)
            history["train_accuracy"].append(train_accuracy)
            history["train_loss"].append(train_loss)
            history["val_accuracy"].append(val_accuracy)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.4f} - Train Loss: {train_loss:.4f} - Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}"
            )

        # Save history log
        self._save_history(history)

        return history

    def predict(self, X):
        """
        Override the predict method to accept 3D image input and flatten it.
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

    def _save_history(self, history):
        """
        Save the training history to a log file.
        """
        log_dir = f"./logs/{self.abbreviation}/"
        os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history_path = os.path.join(log_dir, f"history-{timestamp}.pkl")

        # Save the history
        with open(history_path, "wb") as history_file:
            pickle.dump(history, history_file)
        print(f"Training history saved to '{history_path}'")
