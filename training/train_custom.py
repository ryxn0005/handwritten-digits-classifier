import tensorflow as tf
from keras.src.metrics import Mean
from keras.src.callbacks import EarlyStopping
import numpy as np
import time
from utilities.data_modification.augmentation import apply_rnd_augs
import datetime
from tqdm import tqdm


class Trainer:
    """
    Trainer class to handle the training loop, model evaluation, and logging for neural network models.

    @Parameters:
    model : tf.keras.Model
        The neural network model to be trained.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer used to update model weights.
    train_X : np.ndarray
        Training data features.
    train_y : np.ndarray
        Training data labels.
    loss_fn : tf.keras.losses.Loss
        Loss function used during training.
    metrics : list
        List of metrics to monitor during training (e.g., accuracy).
    epochs : int, optional, default=10
        Number of training epochs.
    batch_size : int, optional, default=32
        Size of each training batch.
    validation_split : float, optional, default=0.2
        Proportion of training data to be used for validation if no separate validation data is provided.
    val_X : np.ndarray, optional, default=None
        Validation data features.
    val_y : np.ndarray, optional, default=None
        Validation data labels.
    num_augs : int, optional, default=2
        Number of augmentations to apply per image.
    callbacks : list, optional, default=None
        List of callbacks such as EarlyStopping.

    @Attributes:
    train_loss_metric : tf.keras.metrics.Mean
        Tracks the average training loss per epoch.
    val_loss_metric : tf.keras.metrics.Mean
        Tracks the average validation loss per epoch.
    train_summary_writer : tf.summary.SummaryWriter
        Summary writer for training metrics.
    val_summary_writer : tf.summary.SummaryWriter
        Summary writer for validation metrics.
    """

    def __init__(
        self,
        model,
        optimizer,
        train_X,
        train_y,
        loss_fn,
        metrics,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        val_X=None,
        val_y=None,
        num_augs=2,
        callbacks=None,
    ):
        self.model, self.optimizer, self.loss_fn = model, optimizer, loss_fn

        # Check if validation data is provided; otherwise, perform a split
        if val_X is not None and val_y is not None:
            self.train_X, self.train_y = train_X, train_y
            self.val_X, self.val_y = val_X, val_y
        else:
            split_index = int(train_X.shape[0] * (1 - validation_split))
            self.train_X, self.val_X = train_X[:split_index], train_X[split_index:]
            self.train_y, self.val_y = train_y[:split_index], train_y[split_index:]

        # Initialize training parameters and metrics
        self.epochs, self.batch_size, self.num_augs, self.metrics = (
            epochs,
            batch_size,
            num_augs,
            metrics,
        )
        self.train_loss_metric, self.val_loss_metric = (
            Mean(name="train_loss"),
            Mean(name="val_loss"),
        )
        self.callbacks = callbacks if callbacks else []
        for callback in self.callbacks:
            callback.set_model(self.model)

        # Set up logging directories
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(
            f"logs/{model.abbreviation}/train/{current_time}"
        )
        self.val_summary_writer = tf.summary.create_file_writer(
            f"logs/{model.abbreviation}/val/{current_time}"
        )

    def augment_batch(self, x_batch):
        """
        Apply random augmentations to each image in the batch.

        @Usage:
            This function applies a specified number of random augmentations
            to each image in the batch.

        @Parameters:
        x_batch : np.ndarray
            Batch of images.

        @Returns:
        np.ndarray : Augmented batch of images.
        """
        return np.array([apply_rnd_augs(img, self.num_augs) for img in x_batch])

    @tf.function
    def train_step(self, x, y):
        """
        Perform a single training step on a batch of data.

        @Usage:
            Calculates the model's loss on a batch of data, computes gradients, and updates model weights.

        @Parameters:
        x : tf.Tensor
            Batch of input data.
        y : tf.Tensor
            Batch of labels.

        @Returns:
        tf.Tensor : Loss value for the batch.
        """
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits) + sum(self.model.losses)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        for metric in self.metrics:
            metric.update_state(y, logits)
        self.train_loss_metric.update_state(loss_value)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        """
        Perform a single validation step on a batch of data.

        @Usage:
            Calculates the model's loss on a batch of validation data and updates validation metrics.

        @Parameters:
        x : tf.Tensor
            Batch of input data.
        y : tf.Tensor
            Batch of labels.

        @Returns:
        tf.Tensor : Validation loss value for the batch.
        """
        val_logits = self.model(x, training=False)
        val_loss_value = self.loss_fn(y, val_logits)
        for metric in self.metrics:
            metric.update_state(y, val_logits)
        return val_loss_value

    def log_metrics(self, writer, metrics, step):
        """
        Log training and validation metrics for each epoch.

        @Usage:
            Writes scalar summaries for specified metrics to the provided writer for TensorBoard visualization.

        @Parameters:
        writer : tf.summary.SummaryWriter
            Summary writer for logging metrics.
        metrics : dict
            Dictionary containing metric names and their values.
        step : int
            Training step or epoch for logging.
        """
        with writer.as_default():
            for name, result in metrics.items():
                tf.summary.scalar(name, result, step=step)

    def reset_metrics(self):
        """
        Reset all training and validation metrics.

        @Usage:
            Resets metric states at the beginning of each epoch to avoid cumulative values.

        @Parameters:
        None

        @Returns:
        None
        """
        for metric in self.metrics + [self.train_loss_metric, self.val_loss_metric]:
            metric.reset_state()

    def train(self):
        """
        Execute the training loop over a specified number of epochs.

        @Usage:
            Handles data shuffling, batch training, validation, metric logging, and early stopping.

        @Parameters:
        None

        @Returns:
        tuple : float
            Average training loss, training accuracy, validation loss, and validation accuracy.
        """
        # Initialize lists to store losses and accuracies for averaging
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        for callback in self.callbacks:
            callback.on_train_begin()

        for epoch in range(self.epochs):
            start_time = time.time()
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # Shuffle the training data at the beginning of each epoch
            indices = np.arange(len(self.train_X))
            np.random.shuffle(indices)
            self.train_X, self.train_y = self.train_X[indices], self.train_y[indices]

            # Training loop
            for i in tqdm(
                range(0, len(self.train_X), self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="batch",
            ):
                batch_X = tf.convert_to_tensor(
                    self.augment_batch(self.train_X[i : i + self.batch_size]),
                    dtype=tf.float32,
                )
                batch_y = tf.convert_to_tensor(
                    self.train_y[i : i + self.batch_size], dtype=tf.int32
                )
                self.train_step(batch_X, batch_y)

            # Calculate average train loss and accuracy for the epoch
            train_loss = self.train_loss_metric.result()
            train_accuracy = self.metrics[
                0
            ].result()  # Assuming the first metric is accuracy
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_metrics_results = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
            }
            self.reset_metrics()

            # Validation loop
            for j in range(0, len(self.val_X), self.batch_size):
                val_loss_value = self.test_step(
                    tf.convert_to_tensor(
                        self.val_X[j : j + self.batch_size], dtype=tf.float32
                    ),
                    tf.convert_to_tensor(
                        self.val_y[j : j + self.batch_size], dtype=tf.int32
                    ),
                )
                self.val_loss_metric.update_state(val_loss_value)

            # Calculate average val loss and accuracy for the epoch
            val_loss = self.val_loss_metric.result()
            val_accuracy = self.metrics[0].result()
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_metrics_results = {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            self.reset_metrics()

            # Print epoch summary
            print(
                f"epoch {epoch + 1}/{self.epochs} - time: {time.time() - start_time:.2f}s"
            )
            train_metrics_str = [
                f"{name}: {result:.4f}"
                for name, result in train_metrics_results.items()
            ]
            val_metrics_str = [
                f"{name}: {result:.4f}" for name, result in val_metrics_results.items()
            ]
            print(" - ".join(train_metrics_str + val_metrics_str))

            # Log metrics for TensorBoard
            self.log_metrics(self.train_summary_writer, train_metrics_results, epoch)
            self.log_metrics(self.val_summary_writer, val_metrics_results, epoch)

            # Execute callback functions at the end of the epoch
            logs = {**train_metrics_results, **val_metrics_results}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs=logs)

            # Check for early stopping
            if any(
                isinstance(callback, EarlyStopping) and callback.stopped_epoch > 0
                for callback in self.callbacks
            ):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Signal end of training to callbacks
        for callback in self.callbacks:
            callback.on_train_end()

        # Return the overall average metrics
        return (
            np.mean(train_losses),
            np.mean(train_accuracies),
            np.mean(val_losses),
            np.mean(val_accuracies),
        )
