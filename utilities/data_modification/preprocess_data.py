import cv2
import numpy as np
import random


def otsu_threshold(x):
    # Temporarily convert to 8-bit for OpenCV processing
    if x.dtype != np.uint8:
        x_uint8 = (x * 255).astype(np.uint8)
    else:
        x_uint8 = x

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(x_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to float32 for TensorFlow compatibility
    thresholded_float = thresholded.astype(np.float32) / 255.0
    return thresholded_float


def gaussian_blur(x, kernel_size=3):
    return cv2.GaussianBlur(x, (kernel_size, kernel_size), 0)


def median_blur(x, kernel_size=3):
    return cv2.medianBlur(x, kernel_size)


def preprocess_image(x):
    # Randomly select one of the two blurring methods
    blur_method = random.choice([gaussian_blur, median_blur])
    x_blurred = blur_method(x)

    # Apply Otsu's thresholding after blurring
    x_thresholded = otsu_threshold(x_blurred)

    return x_thresholded
