import numpy as np
import cv2
import random
import math


def rotate_image(x, angle=10.0):
    """
    Rotates the input image by a specified angle.

    @Usage:
        This function rotates an image around its center by a specified angle,
        preserving the original dimensions.

    @Parameters:
    x : np.ndarray
        Input image array.
    angle : float, optional, default=10.0
        Angle by which to rotate the image.

    @Returns:
    np.ndarray : Rotated image with the same dimensions as the input.
    """
    height, width = x.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(x, rotation_matrix, (width, height))
    return rotated_image


def flip_image(x, axis=1):
    """
    Flips the image along the specified axis.

    @Usage:
        This function performs a horizontal or vertical flip on the image,
        depending on the axis provided.

    @Parameters:
    x : np.ndarray
        Input image array.
    axis : int, optional, default=1
        Axis along which to flip the image. 1 for horizontal, 0 for vertical.

    @Returns:
    np.ndarray : Flipped image.
    """
    return np.flip(x, axis=axis)


def random_brightness(x, min_brightness=0.8, max_brightness=1.2):
    """
    Adjusts the brightness of the input image by a random factor.

    @Usage:
        Modifies image brightness by scaling pixel values by a random factor
        within a specified range.

    @Parameters:
    x : np.ndarray
        Input image array.
    min_brightness : float, optional, default=0.8
        Minimum brightness scaling factor.
    max_brightness : float, optional, default=1.2
        Maximum brightness scaling factor.

    @Returns:
    np.ndarray : Brightness-adjusted image.
    """
    brightness_factor = np.random.uniform(min_brightness, max_brightness)
    bright_image = np.clip(x * brightness_factor, 0, 255).astype(np.float32)
    return bright_image


def cutout(x, mask_size=5):
    """
    Applies a cutout (masking) transformation on the input image.

    @Usage:
        Adds a black square mask to a random location on the image to
        simulate occlusions.

    @Parameters:
    x : np.ndarray
        Input image array.
    mask_size : int, optional, default=5
        Size of the square mask applied to the image.

    @Returns:
    np.ndarray : Image with a random mask applied.
    """
    height, width = x.shape[:2]

    # Ensure mask_size is not larger than the image dimensions
    mask_size = min(mask_size, height, width)

    # If the mask size is still invalid, return the original image
    if mask_size <= 0:
        return x

    # Randomly choose the location of the mask
    mask_x = np.random.randint(0, width - mask_size + 1)
    mask_y = np.random.randint(0, height - mask_size + 1)

    # Apply the mask
    masked_image = x.copy()
    masked_image[mask_y : mask_y + mask_size, mask_x : mask_x + mask_size] = 0
    return masked_image


def zoom_image(x, zoom_factor=1.1):
    """
    Zooms into the image by a specified zoom factor.

    @Usage:
        Resizes the image by a zoom factor, then crops it back to the original dimensions.

    @Parameters:
    x : np.ndarray
        Input image array.
    zoom_factor : float, optional, default=1.1
        Factor by which to zoom into the image. Greater than 1 zooms in, less than 1 zooms out.

    @Returns:
    np.ndarray : Zoomed image with original dimensions.
    """
    if x is None or x.size == 0:
        raise ValueError("Empty image received in zoom_image")

    height, width = x.shape[:2]
    # Calculate the new dimensions based on the zoom factor
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Resize to the zoomed dimensions
    zoomed_image = cv2.resize(
        x, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # Center-crop the zoomed image back to the original dimensions
    crop_y = (new_height - height) // 2
    crop_x = (new_width - width) // 2
    zoomed_image = zoomed_image[crop_y : crop_y + height, crop_x : crop_x + width]

    return zoomed_image


def invert_colors(x):
    """
    Inverts the colors of the input image.

    @Usage:
        Applies a color inversion, transforming each pixel's value to its complement.

    @Parameters:
    x : np.ndarray
        Input image array.

    @Returns:
    np.ndarray : Color-inverted image.
    """
    return 255 - x


def invert_half_image(x, angle=45.0):
    """
    Inverts colors on one half of the image, determined by a specified angle.

    @Usage:
        Inverts the color of pixels on one half of the image based on a line with a specified angle.

    @Parameters:
    x : np.ndarray
        Input image array.
    angle : float, optional, default=45.0
        Angle in degrees of the line dividing the inverted half from the original half.

    @Returns:
    np.ndarray : Half-inverted image.
    """
    height, width = x.shape[:2]
    inverted_image = x.copy()
    angle_rad = math.radians(angle)
    center_x, center_y = width // 2, height // 2
    slope = math.tan(angle_rad)

    for y in range(height):
        for x_coord in range(width):
            if (y - center_y) < slope * (x_coord - center_x):
                inverted_image[y, x_coord] = 255 - x[y, x_coord]

    return inverted_image


# List of transformations with controlled parameters
transformations = [
    lambda img: rotate_image(img, angle=random.uniform(-10, 10)),
    lambda img: flip_image(img, axis=random.choice([0, 1])),
    lambda img: random_brightness(img, 0.8, 1.2),
    lambda img: cutout(img, mask_size=5),
    lambda img: zoom_image(img, zoom_factor=random.uniform(1.0, 1.1)),
    invert_colors,
    lambda img: invert_half_image(img, angle=random.uniform(0, 180)),
]


def apply_rnd_augs(batch, num_transforms=2):
    """
    Applies a random subset of augmentations to each image in the batch independently.

    @Usage:
        Selects a specified number of random transformations from the transformations list
        and applies them to each image in the batch.

    @Parameters:
    batch : np.ndarray
        Batch of images with shape (batch_size, height, width, channels).
    num_transforms : int, optional, default=2
        Number of random transformations to apply to each image in the batch.

    @Returns:
    np.ndarray : Batch of augmented images.
    """
    augmented_batch = []
    for img in batch:
        # Select random transformations for each image
        selected_transforms = random.sample(transformations, num_transforms)
        augmented_img = img.copy()
        for fn in selected_transforms:
            augmented_img = fn(augmented_img)
        augmented_batch.append(augmented_img)
    return np.array(augmented_batch)
