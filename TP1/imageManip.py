import math
import cv2

import numpy as np

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `cv2.imread()` function - 
          whatch out  for the returned color format ! Check the following link for some fun : 
          https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    # Utilisez cv2.imread - le format RGB doit être retourné
    out = cv2.imread(image_path)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    ### VOTRE CODE ICI - FIN

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float32) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = 0.5 * np.square(image)
    ### VOTRE CODE ICI - FIN

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: see if you can use  the opencv function `cv2.cvtColor()` 
    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT    
    out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ### VOTRE CODE ICI - FIN

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = image.copy()
    i = {'R': 0, 'G': 1, 'B': 2}
    if channel in i:
        # Mettre à zéro le canal spécifié
        out[:, :, i[channel]] = 0
    else:
        raise ValueError("Le canal doit être 'R', 'G' ou 'B'.")
    ### VOTRE CODE ICI - FIN

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    limage = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L, A, B = cv2.split(limage)
    
    if channel == "L":
        out = L
    elif channel == "A":
        out = A
    elif channel == "B":
        out = B
    else:
        raise ValueError("Le canal doit être 'L', 'A' ou 'B'.")
    ### VOTRE CODE ICI - FIN

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    H, S, V = cv2.split(hsv_image)
    
    if channel == "H":
        out = H
    elif channel == "S":
        out = S
    elif channel == "V":
        out = V
    else:
        raise ValueError("Le canal doit être 'H', 'S' ou 'V'.")
    ### VOTRE CODE ICI - FIN

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### VOTRE CODE ICI - DEBUT
    image1_excluded = rgb_exclusion(image1, channel1)
    image2_excluded = rgb_exclusion(image2, channel2)

    left_half_image1 = image1_excluded[:, :image1_excluded.shape[1] // 2]
    right_half_image2 = image2_excluded[:, image2_excluded.shape[1] // 2:]

    out = np.concatenate((left_half_image1, right_half_image2), axis=1)
    ### VOTRE CODE ICI - FIN

    return out


def mix_quadrants(image):
    """
    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    height, width, _ = image.shape

    # Define the four quadrants
    top_left = image[:height//2, :width//2]  # Top-left quadrant
    top_right = image[:height//2, width//2:]  # Top-right quadrant
    bottom_left = image[height//2:, :width//2]  # Bottom-left quadrant
    bottom_right = image[height//2:, width//2:]  # Bottom-right quadrant

    # Apply operations to each quadrant
    top_left = rgb_exclusion(top_left, 'R')  # Remove red channel in top left
    top_right = dim_image(top_right)  # Dim the top right quadrant

    # Brighten the bottom left quadrant using the transformation x_n = x_p^0.5
    bottom_left = np.sqrt(bottom_left)  # Element-wise square root to brighten

    bottom_right = rgb_exclusion(bottom_right, 'R')  # Remove red channel in bottom right

    # Combine the quadrants back into a single image
    top_half = np.concatenate((top_left, top_right), axis=1)
    bottom_half = np.concatenate((bottom_left, bottom_right), axis=1)
    out = np.concatenate((top_half, bottom_half), axis=0)
    ### VOTRE CODE ICI - FIN

    return out
