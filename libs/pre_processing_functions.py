import SimpleITK as sitk
import cv2
import numpy as np
import torch


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
    image_oriented = sitk.DICOMOrient(image, 'LPS')
    image_array = sitk.GetArrayFromImage(image_oriented)

    for slice_index in range(image_oriented.GetDepth()):
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
    if np.any(image_array != 0):
        min_value = np.min(image_array)
        max_value = np.max(image_array)

        normalized_image = (image_array - min_value) / (max_value - min_value)
        return normalized_image
    else:
        return image_array


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


def is_binary(image_array):
    """
    Cheching if the provided image is binary.

    Args:
        image_array: np.ndarray representing the 2D image

    Returns:
        True if the image is binary, False otherwise
    """
    bool_binary = np.all(set(image_array.flatten()) <= set([0, 1]))
    
    return bool_binary


def resize_image(image_array):
    """
    Resizing the input image to the desired size.

    Args:
        image_array: np.ndarray representing the image

    Returns:
        np.ndarray representing the resized image resized to be 256x256
    """
    if is_binary(image_array) == True:
        resized_image = cv2.resize(image_array, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
    else:
        resized_image = cv2.resize(image_array, dsize=(256, 256), interpolation = cv2.INTER_CUBIC)

    return resized_image


def pre_processing(sitk_image):
    """
    Performs the preprocessing operation on the raw data
    
    Args: 
        sitk_image: SimpleITK image object containing an axial scan.

    Returns:
        An Torch tensor of size (1, 256, 256) representing the scan, with gray levels
        normalized in the range [0, 1]. 
    """
    image_array = sitk.GetArrayFromImage(sitk_image)
    
    processed_image = normalize_gray_levels(image_array)
    processed_image = resize_image(processed_image)
    torch_image = torch.from_numpy(processed_image).unsqueeze(0)

    return torch_image 