import os
import random
import SimpleITK as sitk
import torch

from libs.dataset import FLAIRDataset


def test_init_missing_data():
    """
    Tests that the FLAIRDataset class raises a FileNotFoundError when the provided data 
    directory doesn't exist.

    GIVEN: A path that does not exist
    WHEN: It is used to initialize a FLAIRDataset object
    THEN: A FileNotFoundError is raised.
    """
    data_dir = "invalid_path" 
    try:
        FLAIRDataset(data_dir)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_len_dataset(data_dir = "dataset/training"):
    """
    Tests that the length of the dataset matches the number of scans contained
    in both the FLAIR and LESION subdirectories.

    GIVEN: A data directory
    WHEN: An instance of FLAIRDataset is created
    THEN: The lenght of the dataset is equal to the number of scans
    """
    dataset = FLAIRDataset(data_dir)
    flair_data = os.path.join(data_dir, "FLAIR")
    lesion_data = os.path.join(data_dir, "LESION")

    flair_scans = [i for i in os.listdir(flair_data)]
    lesion_scans = [f for f in os.listdir(lesion_data)]

    assert len(dataset) == len(flair_scans)
    assert len(dataset) == len(lesion_scans)


def test_getitem_output(data_dir="dataset/training"):
    """
    Tests that the __getitem__ method returns a tuple containing FLAIR and LESION data 
    for the requested slice (index). 
  
    GIVEN: A slice index 
    WHEN: __getitem__() method is called
    THEN: The output is a tuple containing both FLAIR and LESION
    """
    dataset = FLAIRDataset(data_dir)
    data = dataset[0]
    assert isinstance(data, tuple)
    assert len(data) == 2