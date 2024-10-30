from sklearn import svm
import numpy as np
import os
import pickle
from tqdm import tqdm


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

    def train(self, X, y, batch_size=100):
        """
        Train the SVM model on flattened images with approximate batching.
        """
        X_flat = self._flatten_images(X)
        n_samples = X_flat.shape[0]

        # Initialize history dictionary to record metrics
        history = {"batch": [], "samples_seen": [], "accuracy": []}

        support_vectors = None
        support_labels = None

        # Process each batch
        for batch_idx, start in enumerate(
            tqdm(range(0, n_samples, batch_size), desc="Training SVM", unit="batch")
        ):
            end = min(start + batch_size, n_samples)
            X_batch, y_batch = X_flat[start:end], y[start:end]

            if start == 0:
                # Initial fit on the first batch
                super().fit(X_batch, y_batch)
                support_vectors = self.support_
                support_labels = y[self.support_]
            else:
                # Combine support vectors and fit with the new batch
                X_combined = np.vstack([X_flat[support_vectors], X_batch])
                y_combined = np.hstack([support_labels, y_batch])

                # Fit again with combined support vectors and new batch
                super().fit(X_combined, y_combined)
                # Update support vectors with the latest ones
                support_vectors = self.support_
                support_labels = y_combined[self.support_]

            # Calculate accuracy for the current batch
            batch_acc = super().score(X_batch, y_batch)
            history["batch"].append(batch_idx + 1)
            history["samples_seen"].append(len(y_batch))
            history["accuracy"].append(batch_acc)

            print(f"Batch {batch_idx + 1} - Accuracy: {batch_acc:.4f}")

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
