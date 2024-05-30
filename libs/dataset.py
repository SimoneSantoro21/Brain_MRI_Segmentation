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
    Each subdirectory should be named 'Patient-(patient_id)' and contain two files:

      * FLAIR_rot.nii: The FLAIR-MRI scan in Nifti format.
      * LESION_rot.nii: The corresponding lesion segmentation mask in Nifti format.
    """

    def __init__(self, root_path, test = False):
        """
        Initialize the dataset with patient IDs, scan types and root_path
        of the directory where the images are stored.
        """
        self.root_path = root_path
        self.patient_ids = [i for i in range(1, len(os.listdir(root_path)) + 1)]
        self.scan_types = ["FLAIR", "LESION"]

    
    def __len__(self):
        """
        Returns the total number of patients (data samples)
        """
        return len(self.patient_ids)
    

    def __getitem__(self, index):
        """
        Loads and returns the FLAIR-MRI and LESION data for a specific patient (index).

        Args:
            index = Integer index of the patient in the dataset.

        Returns:
            A dictionary containing flair images and segmentation masks as Pytorch Tensors.
        """
        patient_id = self.patient_ids[index - 1]
        unfiltered_patient_data = self.load_patient_data(patient_id)
        patient_data = self.filter_slices(unfiltered_patient_data)
        print(patient_id)

        return patient_data
    

    def load_patient_data(self, patient_id):
        """
        Loads MRI scans for a specific patientusing SimpleITK

        Args:
            patient_id: Integer representing the patient ID.

        Returns:
            A dictionary containing flair and segmentation masks as Numpy arrays
            or None if the patient data is not found
        """
        patient_folder = os.path.join(self.root_path, "Patient-"+str(patient_id))

        if not os.path.isdir(patient_folder):
            print(f"Patient data not found for ID: {patient_id}")
            return None

        patient_data = {}

        for scan_type in self.scan_types:
            filepath = os.path.join(patient_folder, f"{scan_type}_rot.nii")

            if os.path.isfile(filepath):
                scan_img = sitk.ReadImage(filepath)
                processed_axial_slices = PPF.pre_processing(scan_img) 
                patient_data[scan_type] = processed_axial_slices
            else:
                print(f"Scan file missing for {scan_type} in patient {patient_id}")
                return None

        return patient_data
    
    
    def filter_slices(self, patient_data):
        """
        Filters slices to only include those with non-zero lesions and reassigns the keys.

        Args:
            patient_data: Dictionary containing FLAIR and LESION slices.

        Returns:
            Dictionary with filtered slices containing lesions, with keys starting from 0.
        """
        filtered_slices = {'FLAIR': {}, 'LESION': {}}
        new_slice_index = 0

        for slice_idx, lesion_slice in patient_data['LESION'].items():
            lesion_slice = lesion_slice.clone().detach()  # Convert to torch tensor if not already
            if torch.any(lesion_slice):  # Check if any element is non-zero
                filtered_slices['FLAIR'][new_slice_index] = patient_data['FLAIR'][slice_idx]
                filtered_slices['LESION'][new_slice_index] = lesion_slice
                new_slice_index += 1

        return filtered_slices



            