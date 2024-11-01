from keras.src import Model, layers


class ConvNet(Model):
    """
    ConvNet is a convolutional neural network model designed for image classification tasks.

    @Parameters:
    n_classes : int, optional, default=10
        Number of output classes for classification.
    kwargs : dict
        Additional arguments for the Keras Model initialization.

    @Attributes:
    abbreviation : str
        Abbreviation used to name model files and logs.
    conv1, conv2, conv3, conv4 : tf.keras.layers.Conv2D
        Convolutional layers for feature extraction.
    bn1, bn2, bn3, bn4 : tf.keras.layers.BatchNormalization
        Batch normalization layers to stabilize training.
    relu1, relu2, relu3, relu4 : tf.keras.layers.ReLU
        Activation layers for introducing non-linearity.
    pool1, pool2, pool3 : tf.keras.layers.MaxPooling2D
        Pooling layers to reduce spatial dimensions.
    dropout1, dropout2, dropout3, dropout_fc : tf.keras.layers.Dropout
        Dropout layers to prevent overfitting.
    global_pool : tf.keras.layers.GlobalAveragePooling2D
        Global average pooling layer for dimensionality reduction.
    fc1 : tf.keras.layers.Dense
        Fully connected layer for further feature extraction.
    classifier : tf.keras.layers.Dense
        Output layer with softmax activation for class prediction.
    """

    def __init__(self, n_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.abbreviation = "convnet"

        # ConvNet block 1
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.dropout1 = layers.Dropout(0.2)

        # ConvNet block 2
        self.conv2 = layers.Conv2D(64, (3, 3), padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.dropout2 = layers.Dropout(0.3)

        # ConvNet block 3
        self.conv3 = layers.Conv2D(128, (3, 3), padding="same")
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.dropout3 = layers.Dropout(0.3)

        # ConvNet block 4
        self.conv4 = layers.Conv2D(256, (3, 3), padding="same")
        self.bn4 = layers.BatchNormalization()
        self.relu4 = layers.ReLU()

        # Dropout layer
        self.dropout = layers.Dropout(0.4)

        # Global Average Pooling and Dense layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(128, activation="relu")
        self.dropout_fc = layers.Dropout(0.4)
        self.classifier = layers.Dense(n_classes, activation="softmax")

    def call(self, x):
        """
        Forward pass of the ConvNet model, defining the flow of data through each layer.

        @Usage:
            Passes input data through convolutional, pooling, dropout, and dense layers to produce output predictions.

        @Parameters:
        x : tf.Tensor
            Input tensor with shape (batch_size, height, width, channels).

        @Returns:
        tf.Tensor : Prediction tensor with shape (batch_size, n_classes).
        """
        # ConvNet block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # ConvNet block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # ConvNet block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # ConvNet block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        # Dropout and dense layers
        x = self.dropout(x)
        x = self.global_pool(x)
        x = self.fc1(x)
        x = self.dropout_fc(x)
        return self.classifier(x)
