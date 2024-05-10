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

    assert isinstance(normalized_image, np.ndarray)                           
    assert normalized_image.shape == image.shape                              
    assert np.all((normalized_image >= 0) & (normalized_image <= 1))           


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


def test_is_binary():
    """
    Testing that the function is_binary() identifies binary images 

    GIVEN: a np.ndarray representing the 2D image
    WHEN: it is passed as an argument to is_binary()
    THEN: The output is True if the image is binary, False otherwise
    """
    np.random.seed(10)

    binary_image = np.random.choice([0, 1], size=(256, 256))
    non_binary_image = np.random.rand(256, 256)

    assert PPF.is_binary(binary_image) == True
    assert PPF.is_binary(non_binary_image) == False


def test_resize_image_correct_dimension():
    """
    Testing the output dimension of the function resize_image()

    GIVEN: a np.ndarray representing the 2D image
    WHEN: it is passed as an argument to resize_image()
    THEN: The output is an array with shape (256, 256)
    """
    np.random.seed(10)

    random_image_1 = np.random.rand(500, 500)
    random_image_2 = np.random.rand(100, 100)
    random_image_3 = np.random.rand(700, 100)

    assert PPF.resize_image(random_image_1).shape == (256, 256)
    assert PPF.resize_image(random_image_2).shape == (256, 256)
    assert PPF.resize_image(random_image_3).shape == (256, 256)



def test_resize_binary_image():
    """
    Testing that when i apply resize_image() method on a binary image
    the output is still binary

    GIVEN: a binary np.ndarray containing only 0 and 1
    WHEN: it is passed as an argument to resize_image()
    THEN: the output is a binary ndarray
    """
    np.random.seed(10)

    binary_image_1 = np.random.choice([0, 1], size=(500, 500))
    binary_image_2 = np.random.choice([0, 1], size=(100, 100))
    binary_image_3 = np.random.choice([0, 1], size=(600, 200))

    assert PPF.is_binary(PPF.resize_image(binary_image_1)) == True
    assert PPF.is_binary(PPF.resize_image(binary_image_2)) == True
    assert PPF.is_binary(PPF.resize_image(binary_image_3)) == True




        
