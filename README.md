## Overview

This project is designed for image classification tasks, featuring multiple machine learning and deep learning models, utilities for data preprocessing and augmentation, cross-validation, and evaluation metrics. The project is structured into various modules, each serving a specific purpose.

## Folder Structure

```plaintext
.
├── data/                      # Directory for storing datasets
├── evaluation/                # Evaluation scripts
│   ├── confusion_matrix.py       # Script to plot confusion matrix
│   ├── cross_validation.py       # Script for k-fold cross-validation
│   └── math_metrics.py           # Script for calculating metrics like accuracy, precision, recall, F1-score
├── logs/                      # Directory for saving training logs and metrics
├── model/                     # Directory for saving trained model instances
├── network/                   # Directory to contain model architectures (if any additional models are added)
├── results/                   # Directory to store results for each model (e.g., convnet, rfc, svm, vit)
│   ├── convnet/                  # Results specific to ConvNet model
│   ├── rfc/                      # Results specific to Random Forest model
│   ├── svm/                      # Results specific to SVM model
│   └── vit/                      # Results specific to Vision Transformer model
├── training/                  # Training scripts
│   └── train_custom.py           # Script to handle custom model training
├── utilities/                 # Utility scripts for data loading and modification
│   ├── data_loader/              # Data loading utilities
│   │   └── load_data.py             # Script to load and preprocess data
│   └── data_modification/        # Data augmentation and preprocessing utilities
│       ├── augmentation.py         # Script for data augmentation functions
│       └── preprocess_data.py      # Script for data preprocessing functions
├── .gitattributes              # Git attributes configuration
├── .gitignore                  # Git ignore configuration
└── app.py                      # Main application script
```

## Modules and Descriptions

### 1. `evaluation/`
Contains scripts to evaluate the performance of trained models and perform cross-validation.

- **`confusion_matrix.py`**: 
    - Function to plot a confusion matrix for visualizing classification performance.
    - Key Function:
        - `plot_confusion_matrix(y_true, y_pred, figsize, dpi)`: Computes and displays a confusion matrix.

- **`cross_validation.py`**:
    - Performs k-fold cross-validation on specified models.
    - Key Function:
        - `k_fold_cross_validation(models, X, y, ...)`: Trains and evaluates models using k-fold cross-validation and saves the results.

- **`math_metrics.py`**:
    - Calculates metrics such as accuracy, precision, recall, and F1-score for multiclass classification.
    - Key Function:
        - `calculate_multiclass_metrics(y_true, y_pred, num_classes)`: Computes per-class metrics.

### 2. `training/`
Handles custom training procedures for deep learning models.

- **`train_custom.py`**:
    - Contains the main training loop with support for logging, early stopping, and model saving.
    - Key Class:
        - `Trainer`: Manages model training, validation, and logging.

### 3. `utilities/`
Contains helper modules for data loading, preprocessing, and augmentation.

- **`data_loader/`**:
    - **`load_data.py`**:
        - Loads and preprocesses image data for training and validation.
        - Key Class:
            - `CustomDataLoader`: Handles loading of custom datasets, including resizing and normalization.

- **`data_modification/`**:
    - **`augmentation.py`**:
        - Contains functions for applying random augmentations to training images.
        - Key Functions:
            - `apply_rnd_augs(batch, num_transforms)`: Applies random augmentations to a batch of images.
        
    - **`preprocess_data.py`**:
        - Preprocesses images by applying techniques like thresholding and blurring.
        - Key Functions:
            - `preprocess_image(x)`: Preprocesses an image by blurring and applying thresholding.

### 4. `results/`
Stores results (metrics, logs, etc.) for each model, organized by model type.

- **`convnet/`**, **`rfc/`**, **`svm/`**, **`vit/`**:
    - Separate folders to store specific results (e.g., accuracy logs, saved models) for each type of model.
    
### 5. `model/`
Directory intended for saving trained model instances. Each model can be saved with a unique name based on its type and configuration.

## Key Files

- **`.gitignore`**: Specifies files and directories to ignore in version control.
- **`.gitattributes`**: Configuration file for Git attributes.
- **`app.py`**: Main script to run the application, where you can load datasets, initialize models, and execute training and evaluation workflows.

## Usage

To run the main application, use the following command:

```bash
python -m app
```

This will execute `app.py` as the main script, allowing you to load datasets, initialize models, and execute training or evaluation workflows based on the code in `app.py`.

## Dependencies

Ensure you have the following main dependencies installed:

- `tensorflow`: Deep learning framework.
- `keras`: High-level neural network API.
- `scikit-learn`: Tools for machine learning and evaluation.
- `tqdm`: Progress bar library for Python.
- `matplotlib`: Library for creating visualizations.
- `seaborn`: Statistical data visualization library.
- `opencv-python`: Library for image processing.
- `scikit-image`: Image processing in Python.
- `pillow`: Image processing library.
- `cuda-toolkit`: NVIDIA CUDA toolkit for GPU acceleration.
- `cuml`: GPU-accelerated machine learning algorithms by NVIDIA.
- `cudnn`: NVIDIA's deep neural network library for GPU acceleration.
- `numpy`: Library for numerical computations.

## Example

```python
# Example: Training a ConvNet using custom trainer

from network.convnet import ConvNet
from training.train_custom import Trainer
from keras.src.optimizers import AdamW

# Initialize model and optimizer
model = ConvNet(n_classes=10)
optimizer = AdamW(learning_rate=0.001)

# Define Trainer and start training
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_X=train_data,
    train_y=train_labels,
    loss_fn=your_loss_function,
    metrics=[your_metrics],
    epochs=10,
    batch_size=32,
)
trainer.train()
```
