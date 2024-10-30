import os
import cv2
import numpy as np
from tqdm import tqdm
from keras.src.datasets import mnist
from utilities.data_modification.preprocess_data import preprocess_image
import pickle


class CustomDataLoader:
    def __init__(
        self,
        data_dir: str,
        image_size: int = 28,
    ):
        self.data_dir = data_dir  # Base directory for dataset
        self.image_size = image_size  # Size to resize images
        self.data = []
        self.labels = {}  # Store class-to-label mapping
        self.categories = []  # List of categories

    def load_categories(self):
        """
        Find all dataset subfolders in the data directory and assign categories for each dataset.
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
        Build the training data by recursively finding images in each category.
        """
        for category_path in self.categories:  # Iterate over each full category path (e.g., 'dataset_name/0', 'dataset_name/1', ...)
            full_category_path = os.path.join(
                self.data_dir, category_path
            )  # Full path to each category folder
            class_num = self.labels[
                category_path
            ]  # Numeric label for the category (0-9)

            print(f"Processing category: {category_path}")
            # Recursively find all images in subfolders
            image_files = []
            for root, _, files in os.walk(full_category_path):
                image_files.extend(
                    [
                        os.path.join(root, f)
                        for f in files
                        if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
                    ]
                )

            # Apply tqdm to show the progress of reading images
            for img_path in tqdm(
                image_files, desc=f"Processing {category_path}", leave=False
            ):
                try:
                    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image

                    # Check if the image was successfully loaded
                    if img_array is None:
                        print(
                            f"Warning: Unable to load image {img_path}. Skipping this file."
                        )
                        continue

                    # Check if the image has an alpha channel
                    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = cv2.split(img_array)[-1].astype(
                            np.float32
                        )  # Use alpha channel only
                    else:
                        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    resized_array = cv2.resize(
                        img_array, (self.image_size, self.image_size)
                    )  # Resize image

                    processed_image = preprocess_image(resized_array)

                    self.data.append(
                        [processed_image, class_num]
                    )  # Add image and label to training_data
                except OSError as e:
                    print(f"Error reading image {img_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error with image {img_path}: {e}")

        np.random.shuffle(self.data)

    def load_custom_data(self):
        """
        Convert the custom data into features and labels.
        """
        # Separate features and labels
        X = [img for img, label in self.data]
        y = [label for img, label in self.data]

        # Convert the lists to numpy arrays
        X = np.array(X).reshape(
            -1, self.image_size, self.image_size, 1
        )  # (batch_size, height, width, channels)
        y = np.array(y)

        return X, y

    def load_mnist_data(self):
        """
        Load the MNIST data and combine train and test sets.
        """
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Combine train and test data
        X = np.concatenate((X_train, X_test), axis=0).reshape(
            -1, self.image_size, self.image_size, 1
        )  # (batch_size, height, width, channels)
        y = np.concatenate((y_train, y_test), axis=0)

        return X, y

    def merge_data(self):
        """Merge MNIST data with custom data and return combined features and labels."""

        self.load_categories()
        self.make_training_data()

        # Load MNIST and Custom Data
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

    def load_data(
        self,
        processed_data_path: str = "data/processed/combined_data.pkl",
    ):
        """
        Load combined data from 'data/processed' if it exists. If not, load MNIST and custom data,
        merge them, and save to the specified location.
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
                # Load MNIST and Custom Data, and merge them
                X, y = self.merge_data()

                # Save the combined data for future use
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
            # Load MNIST and Custom Data, and merge them
            X, y = self.merge_data()

            # Save the combined data for future use
            os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
            with open(processed_data_path, "wb") as pickle_out:
                pickle.dump((X, y), pickle_out)
            print(f"Combined data saved to '{processed_data_path}'")

        return X, y
