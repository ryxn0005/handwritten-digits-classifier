import cv2
import numpy as np
import random


def otsu_threshold(x):
    """
    Applies Otsu's thresholding to binarize the image.

    @Usage:
        This function first converts the image to an 8-bit format, applies Otsu's thresholding
        for automatic thresholding, and then converts the result back to a float format.

    @Parameters:
    x : np.ndarray
        Input image array. Expects a grayscale image in float32 format.

    @Returns:
    np.ndarray : Thresholded (binarized) image in float32 format.
    """
    # Temporarily convert to 8-bit for OpenCV processing
    if x.dtype != np.uint8:
        x_uint8 = (x * 255).astype(np.uint8)
    else:
        x_uint8 = x

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(x_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to float32 for compatibility
    thresholded_float = thresholded.astype(np.float32) / 255.0
    return thresholded_float


def gaussian_blur(x, kernel_size=3):
    """
    Applies Gaussian blur to the input image.

    @Usage:
        Smooths the image by applying a Gaussian kernel, which reduces noise and detail.

    @Parameters:
    x : np.ndarray
        Input image array.
    kernel_size : int, optional, default=3
        Size of the kernel to use for blurring. Must be an odd number.

    @Returns:
    np.ndarray : Blurred image.
    """
    return cv2.GaussianBlur(x, (kernel_size, kernel_size), 0)


def median_blur(x, kernel_size=3):
    """
    Applies median blur to the input image.

    @Usage:
        Reduces noise by replacing each pixel's value with the median of neighboring pixels.

    @Parameters:
    x : np.ndarray
        Input image array.
    kernel_size : int, optional, default=3
        Size of the kernel to use for median blurring. Must be an odd number.

    @Returns:
    np.ndarray : Blurred image.
    """
    return cv2.medianBlur(x, kernel_size)


def preprocess_image(x):
    """
    Preprocesses the image by applying a random blur and then thresholding.

    @Usage:
        First, applies either Gaussian or median blurring, chosen at random, to reduce noise.
        Then, applies Otsu's thresholding to binarize the image.

    @Parameters:
    x : np.ndarray
        Input image array.

    @Returns:
    np.ndarray : Preprocessed image after blurring and thresholding.
    """
    # Randomly select one of the two blurring methods
    blur_method = random.choice([gaussian_blur, median_blur])
    x_blurred = blur_method(x)

    # Apply Otsu's thresholding after blurring
    x_thresholded = otsu_threshold(x_blurred)

    return x_thresholded
