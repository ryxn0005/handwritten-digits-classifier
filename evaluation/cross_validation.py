import os
import numpy as np
from training.train_custom import Trainer
from keras.src.optimizers import AdamW, Adam
import pickle
from sklearn.utils import resample


def k_fold_cross_validation(
    models,
    X,
    y,
    loss_fn,
    metrics,
    epochs=10,
    batch_size=32,
    num_augs=2,
    learning_rate=0.001,
    weight_decay=0.0001,
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
    model_scores = {model_name: [] for model_name in models.keys()}

    for fold in range(k):
        # Create validation and training sets for the current fold
        val_start = fold * fold_size
        val_end = val_start + fold_size
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
        y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)

        for model_name, model_fn in models.items():
            # Create a new model for each fold
            model = model_fn()

            if model.abbreviation in {"vit", "convnet"}:
                optimizer = AdamW(
                    learning_rate=learning_rate, weight_decay=weight_decay
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
            else:
                # Down-sample the dataset for classical ML models
                X_train_sampled, y_train_sampled = resample(
                    X_train, y_train, n_samples=20000, random_state=fold
                )
                X_val_sampled, y_val_sampled = resample(
                    X_val,
                    y_val,
                    n_samples=min(20000, len(X_val)),
                    random_state=fold,
                )

                # Train Random Forest or SVM models
                log = model.train(
                    X_train_sampled,
                    y_train_sampled,
                    X_val=X_val_sampled,
                    y_val=y_val_sampled,
                )
                val_accuracy = log["val_accuracy"]

            print(
                f"Fold {fold + 1} - {model_name} - Validation Accuracy: {val_accuracy:.4f}"
            )
            model_scores[model_name].append(val_accuracy)

            # Save the model for the current fold
            model_path = os.path.join(
                save_dir,
                f"{model.abbreviation}/fold/{model.abbreviation}_fold_{fold + 1}.keras"
                if model.abbreviation in {"vit", "convnet"}
                else f"{model.abbreviation}/fold/{model.abbreviation}_fold_{fold + 1}.pkl",
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if model.abbreviation in {"vit", "convnet"}:
                model.save(model_path)
            else:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            print(f"Model for fold {fold + 1} saved at: {model_path}")

    # Calculate and print mean scores for each model
    for model_name, scores in model_scores.items():
        mean_score = np.mean(scores)
        print(f"{model_name} - Mean Validation Accuracy: {mean_score:.4f}")

    return model_scores
