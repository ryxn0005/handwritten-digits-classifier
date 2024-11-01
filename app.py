# Library Imports
import os
import keras
import numpy as np
import pickle
from sklearn.utils import resample
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.metrics import (
    SparseCategoricalAccuracy,
    SparseTopKCategoricalAccuracy,
)
from keras.src.optimizers import AdamW

# Custom Imports
from training.train_custom import Trainer
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.math_metrics import calculate_multiclass_metrics
from evaluation.cross_validation import k_fold_cross_validation
from utilities.data_loader.load_data import CustomDataLoader
from network.convnet import ConvNet
from network.vit import VisionTransformer
from network.random_forest import RandomForestImageClassifier
from network.svm import SVMImageClassifier


def load_data(test_size):
    """
    Load and split data into training and testing sets.

    @Usage:
        Loads raw image data, splits it into train and test sets based on `test_size`.

    @Parameters:
    test_size : float
        Proportion of data to be allocated to the test set.

    @Returns:
    tuple : np.ndarray
        Split datasets (X_train, X_test, y_train, y_test).
    """
    c = CustomDataLoader("./data/raw/", 28)
    X, y = c.load_data()
    split_index = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


def specify_hyperparameters(model_type):
    """
    Prompt user to specify model hyperparameters.

    @Usage:
        Allows user to input hyperparameters; provides defaults based on model type.

    @Parameters:
    model_type : str
        Type of model for which hyperparameters are required (e.g., 'svm', 'rfc', 'kfold').

    @Returns:
    dict : Hyperparameter configuration based on user input or defaults.
    """
    # Prompt the user to specify hyperparameters with defaults
    if model_type in {"svm", "rfc"}:
        return {}
    elif model_type == "kfold":
        # Obtain parameters specific to K-Fold Cross Validation
        learning_rate = float(input("Enter learning rate (default 0.001): ") or 0.001)
        weight_decay = float(input("Enter weight decay (default 0.0001): ") or 0.0001)
        batch_size = int(input("Enter batch size (default 256): ") or 256)
        num_epochs = int(input("Enter number of epochs (default 200): ") or 200)
        num_augs = int(input("Enter number of augmentations (default 4): ") or 4)
        return {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_augs": num_augs,
        }

    else:
        # Obtain general parameters for training models
        learning_rate = float(input("Enter learning rate (default 0.001): ") or 0.001)
        weight_decay = float(input("Enter weight decay (default 0.0001): ") or 0.0001)
        batch_size = int(input("Enter batch size (default 256): ") or 256)
        num_epochs = int(input("Enter number of epochs (default 200): ") or 200)
        validation_split = float(input("Enter validation split (default 0.3): ") or 0.3)
        num_augs = int(input("Enter number of augmentations (default 4): ") or 4)
        return {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "validation_split": validation_split,
            "num_augs": num_augs,
        }


def setup_model(model_type, hyperparameters):
    """
    Configure the specified model type with relevant hyperparameters.

    @Usage:
        Loads or creates the chosen model type, setting up hyperparameters and callbacks.

    @Parameters:
    model_type : str
        Type of model to set up (e.g., 'vit', 'convnet', 'rfc', 'svm').
    hyperparameters : dict
        Configuration for hyperparameters such as learning rate, batch size, etc.

    @Returns:
    tuple : Configured model, optimizer, and callbacks.
    """
    optimizer = None
    if model_type == "vit":
        saved_model_path = "./model/vit/vit.keras"
        if os.path.isfile(saved_model_path):
            choice = (
                input("Saved Vision Transformer model found. Load saved model? (y/n): ")
                .strip()
                .lower()
            )
            if choice == "y":
                model = keras.models.load_model(saved_model_path)
            else:
                model = (
                    VisionTransformer(
                        img_size=28,
                        patch_size=7,
                        embed_dim=64,
                        depth=6,
                        n_heads=4,
                        mlp_ratio=4,
                        qkv_bias=True,
                        p=0,
                        attn_p=0,
                        n_classes=10,
                    ),
                )
        else:
            model = VisionTransformer(
                img_size=28,
                patch_size=7,
                embed_dim=64,
                depth=6,
                n_heads=4,
                mlp_ratio=4,
                qkv_bias=True,
                p=0,
                attn_p=0,
                n_classes=10,
            )
            optimizer = AdamW(
                learning_rate=hyperparameters.get("learning_rate"),
                weight_decay=hyperparameters.get("weight_decay"),
            )
    elif model_type == "convnet":
        saved_model_path = "./model/convnet/convnet.keras"
        if os.path.isfile(saved_model_path):
            choice = (
                input("Saved ConvNet model found. Load saved model? (y/n): ")
                .strip()
                .lower()
            )
            if choice == "y":
                model = keras.models.load_model(saved_model_path)
            else:
                model = ConvNet(n_classes=10)
        else:
            model = ConvNet(n_classes=10)
        optimizer = AdamW(
            learning_rate=hyperparameters.get("learning_rate"),
            weight_decay=hyperparameters.get("weight_decay"),
        )
    elif model_type == "rfc":
        saved_model_path = "./model/rfc/rfc.pkl"
        if os.path.isfile(saved_model_path):
            choice = (
                input("Saved Random Forest model found. Load saved model? (y/n): ")
                .strip()
                .lower()
            )
            if choice == "y":
                model = pickle.load(open(saved_model_path, "rb"))
            else:
                model = RandomForestImageClassifier(
                    n_estimators=50,
                    max_depth=20,
                    max_features="sqrt",
                )
        else:
            model = RandomForestImageClassifier(
                n_estimators=50,
                max_depth=20,
                max_features="sqrt",
            )
        optimizer = None  # Not needed for non-deep learning models
    elif model_type == "svm":
        saved_model_path = "./model/svm/svm.pkl"
        if os.path.isfile(saved_model_path):
            choice = (
                input("Saved SVM model found. Load saved model? (y/n): ")
                .strip()
                .lower()
            )
            if choice == "y":
                model = pickle.load(open(saved_model_path, "rb"))
            else:
                model = SVMImageClassifier()
        else:
            model = SVMImageClassifier()
        optimizer = None  # Not needed for non-deep learning models
    else:
        raise ValueError("Invalid model type selected")

    # Set up callbacks if deep learning models are used
    if model_type in {"vit", "convnet"}:
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                mode="min",
                min_delta=0.001,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=saved_model_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
        ]
    else:
        callbacks = None

    return model, optimizer, callbacks


def train_model(model, optimizer, callbacks, hyperparameters, X_train, y_train):
    """
    Train the specified model using provided data and hyperparameters.

    @Usage:
        Initializes and trains the model using the Trainer class if applicable, saves non-deep learning models.

    @Parameters:
    model : object
        The model instance to be trained.
    optimizer : object
        Optimizer configured for deep learning models.
    callbacks : list
        List of callbacks for monitoring model training.
    hyperparameters : dict
        Dictionary containing model training parameters (epochs, batch size, etc.).
    X_train, y_train : np.ndarray
        Training data and labels.
    """
    if hasattr(model, "abbreviation") and model.abbreviation in {"vit", "convnet"}:
        # Train deep learning models with Trainer class
        trainer = Trainer(
            model=model,
            train_X=X_train,
            train_y=y_train,
            optimizer=optimizer,
            loss_fn=SparseCategoricalCrossentropy(),
            metrics=[
                SparseCategoricalAccuracy(name="acc"),
                SparseTopKCategoricalAccuracy(5, name="top_5_acc"),
            ],
            validation_split=hyperparameters.get("validation_split"),
            epochs=hyperparameters.get("num_epochs"),
            batch_size=hyperparameters.get("batch_size"),
            num_augs=hyperparameters.get("num_augs"),
            callbacks=callbacks,
        )
        trainer.train()
    else:
        # Train and save RandomForest and SVM models
        # Resampled datasets for non deep learning models
        X_train_sampled, y_train_sampled = resample(
            X_train, y_train, n_samples=20000, random_state=42
        )

        model.train(
            X_train_sampled,
            y_train_sampled,
        )
        # Save the model for RandomForest and SVM
        save_path = f"./model/{model.abbreviation}/{model.abbreviation}.pkl"
        os.makedirs(
            os.path.dirname(save_path), exist_ok=True
        )  # Ensure directory exists
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")


def log_metrics_to_file(model, metrics):
    """
    Log model performance metrics to a file.

    @Usage:
        Writes metrics to a file for each class, logging accuracy, precision, recall, etc.

    @Parameters:
    model : object
        The model instance whose abbreviation is used to generate file paths.
    metrics : dict
        Dictionary containing calculated metrics for each class.
    """
    # Define the directory and file path
    directory = f"./logs/metrics/{model.abbreviation}"
    file_path = f"{directory}/{model.abbreviation}.txt"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Open the file in write mode
    with open(file_path, "w") as f:
        # Write metrics for each class
        for class_label, class_metrics in metrics.items():
            f.write(f"Class {class_label} metrics:\n")
            for metric_name, metric_value in class_metrics.items():
                f.write(f"  {metric_name}: {metric_value:.4f}\n")
            f.write("\n")


def main():
    """
    Main function to load data, train models, and evaluate performance.

    @Usage:
        Provides a menu interface for model training, loading, and evaluation.
    """
    X_train, X_test, y_train, y_test = None, None, None, None
    while True:
        print("\nMain Menu:")
        print("1. Load Training Data")
        print("2. Train Model")
        print("3. Perform K-Fold Cross Validation")
        print("0. Exit")

        choice = int(input("Select an option (0-3): "))

        if choice == 1:
            # Load data into train/test sets
            X_train, X_test, y_train, y_test = load_data(test_size=0.3)
            print("Data loaded successfully!")
        elif choice == 2:
            # Train model based on user choice
            if X_train is None or X_test is None or y_train is None or y_test is None:
                print("Please load the training data first.")
                continue
            print("Training Menu:")
            print("1. Train Vision Transformer")
            print("2. Train ConvNet")
            print("3. Train Random Forest")
            print("4. Train SVM")
            option = int(input("Pick an option (1 - 4): "))
            model_type = {1: "vit", 2: "convnet", 3: "rfc", 4: "svm"}.get(option)

            if model_type:
                hyperparameters = specify_hyperparameters(model_type)
                model, optimizer, callbacks = setup_model(model_type, hyperparameters)
                train_model(
                    model, optimizer, callbacks, hyperparameters, X_train, y_train
                )
            else:
                print("Invalid option selected.")
        elif choice == 3:
            # Perform K-Fold Cross Validation
            if X_train is None or X_test is None or y_train is None or y_test is None:
                print("Please load the training data first.")
                continue

            print("Specify hyperparameters for K-Fold Cross Validation:")
            n_folds = int(input("Enter number of folds (default 10): ") or 10)
            hyperparameters = specify_hyperparameters("kfold")

            models = {
                "Vision Transformer": lambda: VisionTransformer(
                    img_size=28,
                    patch_size=7,
                    embed_dim=64,
                    depth=6,
                    n_heads=4,
                    mlp_ratio=4,
                    qkv_bias=True,
                    p=0,
                    attn_p=0,
                    n_classes=10,
                ),
                "ConvNet": lambda: ConvNet(n_classes=10),
            }
            model_scores = k_fold_cross_validation(
                models=models,
                X=np.concatenate([X_train, X_test], axis=0),
                y=np.concatenate([y_train, y_test], axis=0),
                loss_fn=SparseCategoricalCrossentropy(),
                metrics=[
                    SparseCategoricalAccuracy(name="acc"),
                    SparseTopKCategoricalAccuracy(5, name="top_5_acc"),
                ],
                epochs=hyperparameters.get("num_epochs"),
                batch_size=hyperparameters.get("batch_size"),
                num_augs=hyperparameters.get("num_augs"),
                callbacks=None,
                k=n_folds,
                save_dir="./model",
            )
        elif choice == 4:
            # Evaluate model performance and log metrics
            print(
                "1. Vision Transformer (ViT)\t2. ConvNet\t3. Random Forest (RFC)\t4. SVM"
            )
            while True:
                try:
                    option = int(input("Choose a model to evaluate (1-4): "))
                    if option in {1, 2, 3, 4}:
                        break
                    else:
                        print("Invalid option. Please enter a number between 1 and 4.")
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 4.")

            # Define model paths for each option
            model_paths = {
                1: "./model/vit/vit.keras",
                2: "./model/convnet/convnet.keras",
                3: "./model/rfc/rfc.pkl",
                4: "./model/svm/svm.pkl",
            }

            if option in {1, 2}:
                # Load Keras model (ViT or ConvNet)
                model_path = model_paths.get(option)
                if model_path and os.path.isfile(model_path):
                    model = keras.models.load_model(model_path)
                    if X_test is not None:
                        y_pred = model.predict(X_test)
                        y_pred_classes = np.argmax(y_pred, axis=1)
                    else:
                        print("Test data not loaded. Returning to main menu.")
                        return
                else:
                    print("Model file not found. Returning to main menu.")
                    return

            elif option in {3, 4}:
                # Load non-deep learning models (RFC or SVM)
                model_path = model_paths.get(option)
                if model_path and os.path.isfile(model_path):
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    if X_test is not None:
                        # Flatten images if needed before making predictions
                        if len(X_test.shape) == 4:
                            n_samples, height, width, channels = X_test.shape
                            X_test_flat = X_test.reshape(
                                n_samples, height * width * channels
                            )
                        else:
                            X_test_flat = X_test
                        y_pred_classes = model.predict(X_test_flat)
                        y_pred_classes = y_pred_classes.to_output("numpy")
                    else:
                        print("Test data not loaded. Returning to main menu.")
                        return
                else:
                    print("Model file not found. Returning to main menu.")
                    return

            else:
                print("Invalid option selected. Returning to main menu.")
                return

            # Plot confusion matrix and calculate metrics if the model is successfully loaded
            plot_confusion_matrix(y_test, y_pred_classes)

            # Calculate accuracy, precision, recall and f1-score of each number for the model
            metrics = calculate_multiclass_metrics(y_test, y_pred_classes)
            log_metrics_to_file(model, metrics)

        elif choice == 0:
            print("Exiting the program.")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
