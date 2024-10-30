from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle
from tqdm import tqdm


class RandomForestImageClassifier(RandomForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.abbreviation = "rfc"

    def _flatten_images(self, X):
        """
        Flattens each image from 2D to 1D if not already flattened. Expects input X of shape (n_samples, height, width, channels).
        """
        # Check if X is already flattened
        if len(X.shape) == 2:  # Already flattened
            return X
        elif len(X.shape) == 4:  # Needs flattening
            n_samples, height, width, in_chans = X.shape
            return X.reshape(n_samples, height * width * in_chans)
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")

    def train(self, X, y, batch_size):
        """
        Fit the model on data in batches and track training history.
        """
        X_flat = self._flatten_images(X)
        n_samples = X_flat.shape[0]

        # Initialize history dictionary to record metrics
        history = {"batch": [], "samples_seen": [], "accuracy": []}

        # Accumulate batches for full training
        accumulated_X = []
        accumulated_y = []

        for batch_idx, start in enumerate(
            tqdm(
                range(0, n_samples, batch_size),
                desc="Training RandomForest",
                unit="batch",
            )
        ):
            end = min(start + batch_size, n_samples)
            X_batch, y_batch = X_flat[start:end], y[start:end]

            # Accumulate data
            accumulated_X.append(X_batch)
            accumulated_y.append(y_batch)

            # Combine accumulated data
            X_combined = np.vstack(accumulated_X)
            y_combined = np.hstack(accumulated_y)

            # Re-train the model with the accumulated data
            super().fit(X_combined, y_combined)

            # Calculate accuracy for the current batch
            accuracy = super().score(X_combined, y_combined)
            history["batch"].append(batch_idx + 1)
            history["samples_seen"].append(len(y_combined))
            history["accuracy"].append(accuracy)

            print(f"Batch {batch_idx + 1} - Accuracy: {accuracy:.4f}")

        return history

    def predict(self, X):
        """
        Override the predict method to accept 3D image input and flatten it.
        """
        X_flat = self._flatten_images(X)  # Flatten the 2D images to 1D
        return super().predict(X_flat)

    def save(self, save_dir: str = "./model/"):
        # Define the path based on model abbreviation
        model_path = os.path.join(save_dir, self.abbreviation)
        os.makedirs(model_path, exist_ok=True)  # Ensure directory exists

        # Save the data using the model abbreviation for the filename
        processed_data_path = os.path.join(model_path, f"{self.abbreviation}.pkl")

        # Save the model
        pickle_out = open(processed_data_path, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()
        print(f"Combined data saved to '{processed_data_path}'")
