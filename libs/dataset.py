import os
import SimpleITK as sitk
from torch.utils.data.dataset import Dataset
import torch

import libs.pre_processing_functions as PPF

class FLAIRDataset(Dataset):
    """
    PyTorch dataset class for FLAIR-MRI scans and corresponding segmented 
    image (LESION) with SimpleITK loading.

    This class expects a specific data organization within the provided root directory. 
    Inside the data directory there should be two folders: "FLAIR" and "LESION":

      * FLAIR: Containing all the 2D axial scans in Nifti format.
      * LESION: The corresponding lesion segmentation masks in Nifti format.
    """

    def __init__(self, root_path, test = False):
        """
        Initialize the dataset with scan indexes, scan types and root_path
        of the directories where the images are stored.
        """
        self.root_path = root_path
        self.FLAIR_path = os.path.join(root_path, "FLAIR") 
        self.LESION_path = os.path.join(root_path, "LESION") 
        self.scan_indexes = [i for i in range(0, len(os.listdir(self.FLAIR_path)))]
        self.scan_types = ["FLAIR", "LESION"]

    
    def __len__(self):
        """
        Returns the total number of FLAIR images (data samples)
        """
        return len(self.scan_indexes)
    

    def __getitem__(self, index):
        """
        Loads and returns the FLAIR-MRI and LESION data for a specific index.

        Args:
            index = Integer index of the scan.

        Returns:
            A tuple containing flair scan and the corresponding segmentation mask 
            as Pytorch Tensors.
        """
        scan_data = self.load_patient_data(index)

        return scan_data["FLAIR"], scan_data["LESION"]


    def load_patient_data(self, index):
        """
        Loads MRI scans for a specific patientusing SimpleITK

        Args:
            index: Integer representing the patient ID.

        Returns:
            A dictionary containing flair and segmentation masks as Torch tensors
            or None if the patient data is not found
        """
        scan_data = {}

        for scan_type in self.scan_types:
            scan_path = os.path.join(self.root_path, scan_type, (f"{scan_type}_{index}.nii"))

            if os.path.isfile(scan_path):
                scan_img = sitk.ReadImage(scan_path)
                processed_img = PPF.pre_processing(scan_img) 
                scan_data[scan_type] = processed_img
            else:
                print(f"Scan file missing for {scan_type} index {index}")
                return None

        return scan_data


            