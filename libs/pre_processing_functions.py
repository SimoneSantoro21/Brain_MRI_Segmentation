import SimpleITK as sitk
import numpy as np


def get_axial_slices(image):
    """
    Obtain all the axial slices form a 3D .nii image.

    Args:
        image: SimpleITK image object

    Returns:
        A dictionary where the keys are the slice number and the values
        are np.ndarray containing the images.
    """

    slices = {}
    image_array = sitk.GetArrayFromImage(image)

    for slice_index in range(image.GetDepth()):
        slice_array = image_array[slice_index]
        slices[slice_index] = slice_array

    return slices


def normalize_gray_levels(image_array):
    """
    Normalize image gray levels between 0 and 1.

    Args:
        image_array: np.ndarray representing the 2D image
    
    Returns:
        np.ndarray representing the same image but with normalized gray levels
    """
    min_value = np.min(image_array)
    max_value = np.max(image_array)

    normalized_image = (image_array - min_value) / (max_value - min_value)

    return normalized_image


def adjust_image_spacing(image):
    """
    Set the spacing of a 2D image to be isotropic (1, 1).

    Args:
        image: SimpleITK Image object representing the 2D image

    Returns:
        SimpleITK Image object with the adjusted spacing
    """
    desired_spacing = (1, 1)
    image.SetSpacing(desired_spacing)

    return None


def resize_image(image_array):
    """
    Resizing the input image to the desired size.

    Args:
        image_array: np.ndarray representing the image

    Returns:
        np.nd
    """

     



    