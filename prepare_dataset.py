import os
import SimpleITK as sitk
import numpy as np
import random


def get_axial_slices(flair_image, lesion_image):
    """
    Extracts and processes axial slices from FLAIR and lesion images containing lesions.

    Args:
        flair_image (SimpleITK.Image): The FLAIR image in SimpleITK format.
        lesion_image (SimpleITK.Image): The lesion segmentation image in SimpleITK format.

    Returns:
        tuple: A tuple containing two lists:
            - flair_slices: A list of SimpleITK 2D images representing axial slices of the FLAIR image that contain lesions.
            - lesion_slices: A list of SimpleITK 2D images representing the corresponding lesion segmentations for each FLAIR slice.
    """
    flair_oriented = sitk.DICOMOrient(flair_image, 'LPS')
    lesion_oriented = sitk.DICOMOrient(lesion_image, 'LPS')
    
    flair_array = sitk.GetArrayFromImage(flair_oriented)
    lesion_array = sitk.GetArrayFromImage(lesion_oriented)
    
    spacing = flair_image.GetSpacing()
    origin = flair_image.GetOrigin()
    direction = flair_image.GetDirection()

    # Extract the 2D direction matrix
    direction_2d = [direction[0], direction[1], direction[3], direction[4]]

    flair_slices = []
    lesion_slices = []

    for slice_index in range(flair_oriented.GetDepth()):
        if np.any(lesion_array[slice_index, :, :] == 1):
            # Process FLAIR slice
            flair_slice = flair_array[slice_index, :, :]
            flair_slice_sitk = sitk.GetImageFromArray(flair_slice)

            flair_slice_sitk.SetSpacing(spacing[1:])
            flair_slice_sitk.SetOrigin(origin[1:])
            flair_slice_sitk.SetDirection(direction_2d)

            flair_slices.append(flair_slice_sitk)

            # Process LESION slice
            lesion_slice = lesion_array[slice_index, :, :]
            lesion_slice_sitk = sitk.GetImageFromArray(lesion_slice)

            # Manually set the metadata for the 2D slice
            lesion_slice_sitk.SetSpacing(spacing[1:])
            lesion_slice_sitk.SetOrigin(origin[1:])
            lesion_slice_sitk.SetDirection(direction_2d)

            lesion_slices.append(lesion_slice_sitk)

    return flair_slices, lesion_slices


def count_slices(input_dir, patient_indices):
    """
    Counts the number of axial slices containing lesions for each specified patient.

    Args:
        input_dir (str): Path to the directory containing patient folders.
        patient_indices (list): List of patient indices (e.g., [10, 24, 36]).

    Returns:
        dict: A dictionary where keys are patient IDs ("Patient-X") and values are the number of lesion-containing slices.
    """    
    num_slices_per_patient = {}
    for patient_index in patient_indices:
        patient_folder = f"Patient-{patient_index}"
        patient_path = os.path.join(input_dir, patient_folder)
        if os.path.isdir(patient_path):
            # Construct the paths for FLAIR and LESION files
            flair_file = os.path.join(patient_path, 'FLAIR_rot.nii')
            lesion_file = os.path.join(patient_path, 'LESION_rot.nii')

            # Process files
            if os.path.isfile(flair_file) and os.path.isfile(lesion_file):
                flair_image = sitk.ReadImage(flair_file)
                lesion_image = sitk.ReadImage(lesion_file)
                flairs, _ = get_axial_slices(flair_image, lesion_image)
                num_slices_per_patient[f"Patient-{patient_index}"] = len(flairs)

    return num_slices_per_patient


def split_patient_data(patient_data, train_fract):
    """
    Randomly splits patient data into training and validation sets based on a specified fraction.

    Args:
        patient_data (dict): A dictionary mapping patient IDs to the number of slices they have.
        train_fract (float): Fraction of patients to include in the training set (0.0 to 1.0).

    Returns:
        tuple: A tuple containing two lists:
            - train_patients: List of patient IDs for the training set.
            - val_patients: List of patient IDs for the validation set.

    Prints:
        - Lists of training and validation patients.
        - Average number of images per patient in each set, along with the total number of images.
    """
    random.seed(42)  

    patients = list(patient_data.keys())
    random.shuffle(patients)

    split_index = int(train_fract * len(patients))
    train_patients = patients[:split_index]
    val_patients = patients[split_index:]

    print("Training patients:", train_patients)
    print("Validation patients:", val_patients)

    train_images = sum(patient_data[patient] for patient in train_patients)
    val_images = sum(patient_data[patient] for patient in val_patients)

    train_avrg_patient_images = train_images / (len(train_patients))
    val_avrg_patient_images = val_images / (len(val_patients))
    print(train_avrg_patient_images, train_images)
    print(val_avrg_patient_images, val_images)

    return train_patients, val_patients


def save_dataset(patients, RAW_DATA_PATH, DATA_PATH):
    """
    Saves axial slices containing lesions from specified patients into separate FLAIR and LESION directories.

    Args:
        patients (list): List of patient IDs to process.
        RAW_DATA_PATH (str): Path to the raw data directory containing patient folders.
        DATA_PATH (str): Path to the destination directory where FLAIR and LESION folders will be created.

    Returns:
        None

    Prints:
        - Progress messages as each slice is saved.
        - Error messages if patient folders or files are not found.
    """    
    flair_dir = os.path.join(DATA_PATH, 'FLAIR')
    lesion_dir = os.path.join(DATA_PATH, 'LESION')

    os.makedirs(flair_dir, exist_ok=True)
    os.makedirs(lesion_dir, exist_ok=True)

    renaming_index = 0

    for patient_index in patients:
        patient_folder = f"{patient_index}"
        patient_path = os.path.join(RAW_DATA_PATH, patient_folder)
        if os.path.isdir(patient_path):
            # Construct the paths for FLAIR and LESION files
            flair_file = os.path.join(patient_path, 'FLAIR_rot.nii')
            lesion_file = os.path.join(patient_path, 'LESION_rot.nii')

            # Process files
            if os.path.isfile(flair_file) and os.path.isfile(lesion_file):
                flair_image = sitk.ReadImage(flair_file)
                lesion_image = sitk.ReadImage(lesion_file)
                slices = get_axial_slices(flair_image, lesion_image)
                flair_slices = slices[0]
                lesion_slices = slices[1]
                
                for slice_index in range(len(flair_slices)):
                    flair_save_path = os.path.join(flair_dir, f'FLAIR_{slice_index + renaming_index}.nii')
                    flair_slice = flair_slices[slice_index]
                    sitk.WriteImage(flair_slice, flair_save_path)
                    print(f'Saved {flair_save_path}')

                    lesion_save_path = os.path.join(lesion_dir, f'LESION_{slice_index + renaming_index}.nii')
                    lesion_slice = lesion_slices[slice_index]
                    sitk.WriteImage(lesion_slice, lesion_save_path)
                    print(f'Saved {lesion_save_path}')
                renaming_index += slice_index + 1
        else:
            print(f'Patient folder {patient_folder} does not exist')

    return None


if __name__ == '__main__':
    RAW_DATA_PATH = "data_raw"
    RAW_TESTING_PATH = "test_dataset"
    DATA_PATH = "dataset"
    TRAINING_PATH = os.path.join(DATA_PATH, "training")
    VALIDATION_PATH = os.path.join(DATA_PATH, "validation")
    TESTING_PATH = os.path.join(DATA_PATH, "testing")

    accepted_indexes = [10, 24, 36, 5, 11, 25, 37, 50, 12, 26, 38, 51, 13, 27, 39,
                    52, 15, 28, 4, 55, 17, 29, 40, 58, 18, 3, 42, 59, 19, 30,
                    43, 6, 2, 31, 44, 60, 20, 33, 45, 7, 21, 34, 47, 8, 22,
                    35, 49, 9]
    accepted_indexes.sort()

    number_slices = count_slices(RAW_DATA_PATH, accepted_indexes)   #dict containing total filtered slice number per patient
    train_patients, val_patients = split_patient_data(number_slices, train_fract=0.8)   #train_fract = fraction of imgs to put in traning set
    save_dataset(train_patients, RAW_DATA_PATH, TRAINING_PATH)
    save_dataset(val_patients, RAW_DATA_PATH, VALIDATION_PATH)
    
