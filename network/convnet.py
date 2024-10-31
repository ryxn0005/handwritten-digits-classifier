from keras.src import Model, layers


class ConvNet(Model):
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
