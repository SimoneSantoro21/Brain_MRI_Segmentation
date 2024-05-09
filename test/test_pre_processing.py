import random
import SimpleITK as sitk
import numpy as np

import libs.pre_processing_functions as PPF


def test_get_axial_slices():
    """
    Testing that the function get_axial_slices() output is correct.

    GIVEN: a SimpleITK image object
    WHEN: I pass it to the get_axial_slices() function
    THEN: The output is a dictionary containing np.ndarrays of the single slices
    """
    random.seed(10)
    
    for i in range(10): 
        test_image_width = int(random.uniform(1, 10))
        test_image_height = int(random.uniform(1, 10))
        test_image_depth = int(random.uniform(1, 10))
        test_image = sitk.Image(test_image_width, test_image_height, test_image_depth, sitk.sitkUInt8)
        
        slices = PPF.get_axial_slices(test_image)

        assert isinstance(slices, dict)
        assert isinstance(slices[0], np.ndarray)
        assert len(slices) == test_image_depth
        

def test_normalize_image():
    """
    Testing that the output of normalize_gray_levels() is a properly normalized image.

    GIVEN: a np.ndarray representing the 2D image
    WHEN: it is passed as an argument to the normalize_grey_levels() function
    THEN: The output is an np.ndarray containing the same number of elements, ranging in the interval [0, 1]
    """
    np.random.seed(10)

    image = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    normalized_image = PPF.normalize_gray_levels(image.copy())

    assert isinstance(normalized_image, np.ndarray),                            "Output must be a NumPy array."
    assert normalized_image.shape == image.shape,                               "Output shape must match input shape."
    assert np.all((normalized_image >= 0) & (normalized_image <= 1)),           "Values must be between 0 and 1."


def test_adjust_image_spacing():
    """
    Testing that the output of adjust_image_spacing() is a properly spaced image.

    GIVEN: a np.ndarray representing the 2D image
    WHEN: it is passed as an argumento to the adjust_image_spacing() function
    THEN: The image spacing gets adjusted to be (1, 1)
    """
    random.seed(10)

    test_image_width = random.randint(1, 10)
    test_image_height = random.randint(1, 10)
    test_image_spacing = (random.randint(1, 5), random.randint(1, 5))

    test_image = sitk.Image(test_image_width, test_image_height, sitk.sitkUInt8)
    test_image.SetSpacing(test_image_spacing)
    

    PPF.adjust_image_spacing(test_image)

    assert test_image.GetSpacing() == (1, 1)



        
