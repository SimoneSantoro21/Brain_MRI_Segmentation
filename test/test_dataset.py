import os
import random
import SimpleITK as sitk
import torch

from libs.dataset import FLAIRDataset


def test_init_missing_data():
    """
    Tests that the FLAIRDataset class raises a FileNotFoundError when the provided data directory doesn't exist.
    """
    # Simulate missing data directory (assuming data is in the same directory)
    data_dir = "invalid_path"  # Replace with a non-existent directory
    try:
        FLAIRDataset(data_dir)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_len_dataset(data_dir = "data"):
    """
    Tests that the length of the dataset (number of patients) matches the
    number of subdirectories within the "data" directory (excluding non-data folders).

    GIVEN: A data directory where the patient folders start with "Patient-"
    WHEN: An instance of FLAIRDataset is created
    THEN: The lenght of the dataset is equal to the number of patient directories
    """
    dataset = FLAIRDataset(data_dir)

    # Assuming subdirectory names start with "Patient-" (modify if needed)
    patient_folders = [f for f in os.listdir(data_dir) if f.startswith("Patient-")]
    assert len(dataset) == len(patient_folders)


def test_getitem_output(data_dir="data"):
    """
    Tests that the __getitem__ method returns a dictionary containing FLAIR and LESION data 
    for the requested patient (index). 
  
    GIVEN: A patient index 
    WHEN: __getitem__() method is called
    THEN: The output is a dictionary containing both FLAIR and LESION
    """
    dataset = FLAIRDataset(data_dir)
    patient_data = dataset[0]
    assert isinstance(patient_data, dict)
    assert "FLAIR" in patient_data.keys()
    assert "LESION" in patient_data.keys()


def test_getitem_output_filter(data_dir="data"):
    """
    Tests that the __getitem__ method's output only contains scans in which there is at least
    one lesion.

    GIVEN: A patient index
    WHEN: __getitem__() method is called
    THEN: Every LESION scan of the patient contains at least a pixel of value 1
    """
    dataset = FLAIRDataset(data_dir)
    patient_data = dataset[0]
    for lesion_index in range(len(patient_data["LESION"])):
        assert torch.any(patient_data["LESION"][lesion_index])
