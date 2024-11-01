import os
import cv2
import numpy as np
from tqdm import tqdm
from keras.src.datasets import mnist
from utilities.data_modification.preprocess_data import preprocess_image
import pickle


class CustomDataLoader:
    """
    CustomDataLoader class for loading and processing both custom and MNIST datasets,
    allowing them to be merged for training deep learning models.

    @Parameters:
    data_dir : str
        Path to the directory containing the custom dataset.
    image_size : int, optional, default=28
        Target size for resizing images.

    @Attributes:
    data_dir : str
        Base directory for the dataset.
    image_size : int
        Target size for resizing images.
    data : list
        List containing processed images and their corresponding labels.
    labels : dict
        Dictionary mapping class paths to numeric labels.
    categories : list
        List of paths to each category (label) directory within the dataset.
    """

    def __init__(self, data_dir: str, image_size: int = 28):
        self.data_dir = data_dir  # Base directory for dataset
        self.image_size = image_size  # Size to resize images
        self.data = []
        self.labels = {}  # Store class-to-label mapping
        self.categories = []  # List of categories

    def load_categories(self):
        """
        Find all subfolders in the data directory to set up dataset categories and labels.

        @Usage:
            Searches through the data directory to locate subfolders, assigning numeric labels
            based on folder names.

        @Parameters:
        None

        @Returns:
        None
        """
        dataset_subfolders = [
            folder
            for folder in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, folder))
        ]

        for dataset in dataset_subfolders:
            dataset_path = os.path.join(self.data_dir, dataset)
            category_folders = sorted(
                [
                    folder
                    for folder in os.listdir(dataset_path)
                    if os.path.isdir(os.path.join(dataset_path, folder))
                    and folder.isdigit()
                ],
                key=lambda x: int(x),
            )

            # Append categories from each dataset (label remains 0-9)
            self.categories.extend(
                [os.path.join(dataset, category) for category in category_folders]
            )
            # Assign labels simply as 0 to 9 based on folder names
            for category in category_folders:
                self.labels[os.path.join(dataset, category)] = int(category)

        print(f"Found categories: {self.categories}")
        print(f"Labels: {self.labels}")

    def make_training_data(self):
        """
        Build the training data by loading, resizing, and preprocessing images from each category.

        @Usage:
            Recursively searches each category for images, preprocesses each, and adds them to
            the training data.

        @Parameters:
        None

        @Returns:
        None
        """
        for category_path in self.categories:
            full_category_path = os.path.join(self.data_dir, category_path)
            class_num = self.labels[category_path]

            print(f"Processing category: {category_path}")
            image_files = []
            for root, _, files in os.walk(full_category_path):
                image_files.extend(
                    [
                        os.path.join(root, f)
                        for f in files
                        if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
                    ]
                )

            # Apply tqdm to show progress of reading images
            for img_path in tqdm(
                image_files, desc=f"Processing {category_path}", leave=False
            ):
                try:
                    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                    # Skip images that cannot be loaded
                    if img_array is None:
                        print(f"Warning: Unable to load image {img_path}. Skipping.")
                        continue

                    # If image has an alpha channel, use only the alpha channel
                    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = cv2.split(img_array)[-1].astype(np.float32)
                    else:
                        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # Resize and preprocess the image
                    resized_array = cv2.resize(
                        img_array, (self.image_size, self.image_size)
                    )
                    processed_image = preprocess_image(resized_array)

                    self.data.append(
                        [processed_image, class_num]
                    )  # Add processed image and label
                except OSError as e:
                    print(f"Error reading image {img_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error with image {img_path}: {e}")

        np.random.shuffle(self.data)

    def load_custom_data(self):
        """
        Extracts features and labels from custom data.

        @Usage:
            Separates images and their labels into separate arrays.

        @Parameters:
        None

        @Returns:
        tuple : np.ndarray
            Features (X) and labels (y) as separate arrays.
        """
        X = [img for img, label in self.data]
        y = [label for img, label in self.data]

        X = np.array(X).reshape(-1, self.image_size, self.image_size, 1)
        y = np.array(y)

        return X, y

    def load_mnist_data(self):
        """
        Loads the MNIST dataset and combines train and test sets.

        @Usage:
            Loads MNIST data, resizes to the specified image size, and reshapes for model input.

        @Parameters:
        None

        @Returns:
        tuple : np.ndarray
            MNIST data as combined features (X) and labels (y).
        """
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X = np.concatenate((X_train, X_test), axis=0).reshape(
            -1, self.image_size, self.image_size, 1
        )
        y = np.concatenate((y_train, y_test), axis=0)

        return X, y

    def merge_data(self):
        """
        Merges MNIST data with custom data and returns combined features and labels.

        @Usage:
            Loads both custom and MNIST data, merges them, shuffles the combined dataset,
            and returns it.

        @Parameters:
        None

        @Returns:
        tuple : np.ndarray
            Combined dataset with features (X) and labels (y).
        """
        self.load_categories()
        self.make_training_data()

        X_mnist, y_mnist = self.load_mnist_data()
        X_custom, y_custom = self.load_custom_data()

        # Concatenate MNIST and Custom Data
        X = np.concatenate((X_mnist, X_custom), axis=0)
        y = np.concatenate((y_mnist, y_custom), axis=0)

        np.random.seed()

        # Shuffle the combined data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        return X, y

    def load_data(self, processed_data_path: str = "data/processed/combined_data.pkl"):
        """
        Load combined dataset from a processed file, or create and save it if not found.

        @Usage:
            If combined data exists in the specified path, load it. Otherwise, merge MNIST
            and custom data, save, and return it.

        @Parameters:
        processed_data_path : str, optional, default="data/processed/combined_data.pkl"
            Path to save or load the processed combined dataset.

        @Returns:
        tuple : np.ndarray
            Combined features (X) and labels (y) from MNIST and custom data.
        """
        # Check if the processed data file exists
        if os.path.exists(processed_data_path):
            choice = (
                input(
                    f"Processed data found at '{processed_data_path}'.\n"
                    f"Would you like to (L)oad the existing data or (C)reate a new one? [L/C]: "
                )
                .strip()
                .upper()
            )

            if choice == "L":
                print(f"Loading processed data from '{processed_data_path}'...")
                with open(processed_data_path, "rb") as pickle_in:
                    X, y = pickle.load(pickle_in)
            elif choice == "C":
                print("Creating new data by loading and merging raw data...")
                X, y = self.merge_data()
                os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                with open(processed_data_path, "wb") as pickle_out:
                    pickle.dump((X, y), pickle_out)
                print(f"New combined data saved to '{processed_data_path}'")
            else:
                print("Invalid choice. Loading existing data by default.")
                with open(processed_data_path, "rb") as pickle_in:
                    X, y = pickle.load(pickle_in)
        else:
            print("Processed data not found. Loading and merging raw data...")
            X, y = self.merge_data()
            os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
            with open(processed_data_path, "wb") as pickle_out:
                pickle.dump((X, y), pickle_out)
            print(f"Combined data saved to '{processed_data_path}'")

        return X, y
