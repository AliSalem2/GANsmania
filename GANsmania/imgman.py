import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_image(image_path: str) -> np.ndarray:
    """Read an image from a path.

    :param image_path: Read the image from this path
    :return: Image object.
    """
    return cv2.imread(image_path)


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image by a certain angle.

    :param img: Image object.
    :param angle: Image rotation angle.
    :return: Rotated image object.
    """
    h, w = img.shape[:2]

    # Define rotation center (center of image)
    c_x, c_y = w / 2, h / 2
    # Define rotation matrix with X degrees of rotation
    rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)

    # Rotate the image
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated


def rotate_image_fixed_angles(img: np.ndarray):
    """Rotate an image by 90, 180 and 270 degrees.

    Returns a list of the rotated images and a list of angles.

    :param img: Image object.
    :return: Returns a list of images and a list of angles
    """
    rotated_images = []
    angles = []

    # rotate image 90 degrees
    img_90 = cv2.transpose(img)
    img_90 = cv2.flip(img_90, 1)
    rotated_images.append(img_90)
    angles.append(90)

    # rotate image 180 degrees
    img_180 = cv2.flip(img, -1)
    rotated_images.append(img_180)
    angles.append(180)

    # rotate image 270 degrees
    img_270 = cv2.transpose(img)
    img_270 = cv2.flip(img_270, 0)
    rotated_images.append(img_270)
    angles.append(270)

    return rotated_images, angles


def scale_image(img: np.ndarray, factor: float, interpolation) -> np.ndarray:
    """Scale an image"""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation)


def crop_image(img: np.ndarray, x1: int, x2: int, y1: int, y2: int) -> np.ndarray:
    """Crops an image.

    Provide X and Y bounds. y2 > y1 and x2 > x1.

    See provided image for reference (``docs/crop.png``).

    .. image:: docs/crop.png

    :param img: Image object.
    :param x1: Lower X bound
    :param x2: Upper X bound
    :param y1: Lower Y bound
    :param y2: Upper Y bound
    :return: Returns cropped image
    """
    return img[int(y1):int(y2), int(x1):int(x2)]


def is_mostly_black(img: np.ndarray, color_threshold: int, blackness: float) -> bool:
    """
    Detect if image is mostly black.

    :param img: Image object.

    :param color_threshold:
        (0 - 255)
        Everything below this value is considered black.

    :param blackness:
        (0.0 - 1.0)
        Ratio between black and bright pixels.
        An image is considered black if it's ratio of black pixels
        is above the defined blackness value.

    :return:
        Boolean, whether the image is mostly black or not
    """
    # Set all color values below threshold to "True" and the rest to "False"
    binary_img = (img <= color_threshold)

    # Reshape the image so it's basically one long line of lists.
    # We remove one dimension.
    new_shape = (binary_img.shape[0] * binary_img.shape[1], 3)
    reshaped = np.reshape(binary_img, new_shape)

    # Now check which array elements are pure black
    blacks = np.all(reshaped, axis=1)

    # Calculate image dimensions
    num_values = blacks.shape[0]
    zero_values = blacks[blacks[:] == True].shape[0]

    # Finally return if the image is mostly black
    return zero_values > num_values * blackness


def save_image(img: np.ndarray, filepath: str, colorspace: int = 4):
    """
    Save an image object to a file.

    :param img: Image object
    :param filepath: Save image to this path
    :param colorspace: cv2 Colorspace Integer
    """
    plt.imsave(filepath, cv2.cvtColor(img, colorspace))
