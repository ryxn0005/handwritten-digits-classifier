import os
import numpy as np
from training.train import Trainer
from keras.src.optimizers import AdamW, Adam


def k_fold_cross_validation(
    model_fn,
    X,
    y,
    loss_fn,
    metrics,
    epochs=10,
    batch_size=32,
    num_augs=2,
    callbacks=None,
    k=5,
    save_dir="./model",
):
    # Create the directory to save models if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Shuffle the data to ensure randomness
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split data into k folds
    fold_size = len(X) // k
    scores = []

    for fold in range(k):
        # Create validation and training sets for the current fold
        val_start = fold * fold_size
        val_end = val_start + fold_size
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
        y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)

        # Create a new model for each fold
        model = model_fn()

        learning_rate = 0.001

        optimizer = (
            AdamW(learning_rate=learning_rate, weight_decay=0.0001)
            if model.abbreviation == "vit"
            else Adam(learning_rate=learning_rate)
        )

        # Train the model on the current fold
        trainer = Trainer(
            model=model,
            train_X=X_train,
            train_y=y_train,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            val_X=X_val,
            val_y=y_val,
            epochs=epochs,
            batch_size=batch_size,
            num_augs=num_augs,
            callbacks=callbacks,
        )
        train_loss, train_accuracy, val_loss, val_accuracy = trainer.train()
        print(f"Fold {fold + 1} - Validation Accuracy: {val_accuracy:.4f}")
        scores.append(val_accuracy)

        # Save the model for the current fold
        model_path = os.path.join(
            save_dir,
            f"{model.abbreviation}/fold/{model.abbreviation}_fold_{fold + 1}.keras",
        )
        model.save(model_path)
        print(f"Model for fold {fold + 1} saved at: {model_path}")

    return np.mean(scores)
