from utilities.data_loader.load_data import CustomDataLoader
from network.convnet import ConvNet
from network.vit import VisionTransformer
from network.random_forest import RandomForestImageClassifier
from network.svm import SVMImageClassifier
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.metrics import (
    SparseCategoricalAccuracy,
    SparseTopKCategoricalAccuracy,
)
from keras.src.optimizers import AdamW, Adam
from training.train_custom import Trainer
import os
import keras
import numpy as np
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.math_metrics import calculate_multiclass_metrics
from evaluation.cross_validation import k_fold_cross_validation
import pickle


def load_data():
    c = CustomDataLoader("./data/raw/", 28)
    X, y = c.load_data()

    test_size = 0.3
    split_index = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


def specify_hyperparameters(model_type):
    # Prompt the user to specify hyperparameters with defaults
    if model_type in {"svm", "rfc"}:
        batch_size = int(input("Enter batch size (default 256): ") or 256)
        num_epochs = int(input("Enter number of epochs (default 200): ") or 200)
        return {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    else:
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
    """Setup model based on the type and hyperparameters provided."""
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
                model = VisionTransformer()
        else:
            model = VisionTransformer()
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
                model = ConvNet()
        else:
            model = ConvNet()
        optimizer = Adam(learning_rate=hyperparameters.get("learning_rate"))
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
                model = RandomForestImageClassifier()
        else:
            model = RandomForestImageClassifier()
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
        optimizer = None
    else:
        raise ValueError("Invalid model type selected")

    # Set up callbacks if applicable
    if model_type in {"vit", "convnet"}:
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                mode="max",
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
    if hasattr(model, "abbreviation") and model.abbreviation in {"vit", "convnet"}:
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
        model.train(
            X_train,
            y_train,
            batch_size=hyperparameters.get("batch_size"),
            epochs=hyperparameters.get("num_epochs"),
        )
        # Save the model for RandomForest and SVM
        save_path = f"./model/{model.abbreviation}/{model.abbreviation}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")


def main():
    X_train, X_test, y_train, y_test = None, None, None, None

    while True:
        print("\nMain Menu:")
        print("1. Load Training Data")
        print("2. Train Model")
        print("3. Perform K-Fold Cross Validation")
        print("0. Exit")

        choice = int(input("Select an option (0-3): "))

        if choice == 1:
            X_train, X_test, y_train, y_test = load_data()
            print("Data loaded successfully!")
        elif choice == 2:
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
            if X_train is None or X_test is None or y_train is None or y_test is None:
                print("Please load the training data first.")
                continue
            models = {
                "Vision Transformer": VisionTransformer,
                "ConvNet": ConvNet,
                "Random Forest": RandomForestImageClassifier,
                "SVM": SVMImageClassifier,
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
                epochs=10,
                batch_size=32,
                num_augs=2,
                callbacks=None,
                k=5,
                save_dir="./model",
            )
        elif choice == 0:
            print("Exiting the program.")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()


# cross_validation_score = k_fold_cross_validation(
#     VisionTransformer,
#     X=X_train,
#     y=y_train,
#     loss_fn=loss_fn,
#     metrics=[SparseCategoricalAccuracy(name="acc")],
#     epochs=1,
#     batch_size=256,
#     num_augs=2,
#     k=10,
# )
#

# print(cross_validation_score)

# y_pred = model.predict(X_custom)
#
# y_pred_classes = np.argmax(y_pred, axis=1)
#
#
# plot_confusion_matrix(y_custom, y_pred_classes)

# metrics = calculate_multiclass_metrics(y_test, y_pred_classes)

# Print metrics for each class
# for class_label, class_metrics in metrics.items():
#     print(f"Class {class_label} metrics:")
#     for metric_name, metric_value in class_metrics.items():
#         print(f"  {metric_name}: {metric_value:.4f}")
#

# X_train = X_train[:1250]
# X_test = X_test[:1250]
# y_train = y_train[:1250]
# y_test = y_test[:1250]
#
#
# rf = SVMImageClassifier()
#
# rf.train(X_train, y_train, 64)
#
# y_pred = rf.predict(X_test)
#
# print(y_pred)
# print(y_test)
