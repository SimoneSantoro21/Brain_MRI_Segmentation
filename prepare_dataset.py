import os
import SimpleITK as sitk
import numpy as np

accepted_indexes = [10, 24, 36, 5, 11, 25, 37, 50, 12, 26, 38, 51, 13, 27, 39,
                    52, 15, 28, 4, 55, 17, 29, 40, 58, 18, 3, 42, 59, 19, 30,
                    43, 6, 2, 31, 44, 60, 20, 33, 45, 7, 21, 34, 47, 8, 22,
                    35, 49, 9]

accepted_indexes.sort()


def get_axial_slices(flair_image, lesion_image, flair_dir, lesion_dir, current_index):
    """Extracts axial slices from 3D FLAIR and LESION images and saves them as 2D NIfTI images."""
    flair_oriented = sitk.DICOMOrient(flair_image, 'LPS')
    lesion_oriented = sitk.DICOMOrient(lesion_image, 'LPS')
    
    flair_array = sitk.GetArrayFromImage(flair_oriented)
    lesion_array = sitk.GetArrayFromImage(lesion_oriented)
    
    spacing = flair_image.GetSpacing()
    origin = flair_image.GetOrigin()
    direction = flair_image.GetDirection()

    # Extract the 2D direction matrix
    direction_2d = [direction[0], direction[1], direction[3], direction[4]]

    # Save each axial slice as a 2D image if it contains lesions
    for slice_index in range(flair_oriented.GetDepth()):
        if np.any(lesion_array[slice_index, :, :] == 1):
            # Process FLAIR slice
            flair_slice = flair_array[slice_index, :, :]
            flair_slice_sitk = sitk.GetImageFromArray(flair_slice)

            # Manually set the metadata for the 2D slice
            flair_slice_sitk.SetSpacing(spacing[1:])
            flair_slice_sitk.SetOrigin(origin[1:])
            flair_slice_sitk.SetDirection(direction_2d)

            # Construct the new file name for FLAIR
            flair_file = os.path.join(flair_dir, f'FLAIR_{current_index}.nii')
            sitk.WriteImage(flair_slice_sitk, flair_file)
            print(f'Saved {flair_file}')

            # Process LESION slice
            lesion_slice = lesion_array[slice_index, :, :]
            lesion_slice_sitk = sitk.GetImageFromArray(lesion_slice)

            # Manually set the metadata for the 2D slice
            lesion_slice_sitk.SetSpacing(spacing[1:])
            lesion_slice_sitk.SetOrigin(origin[1:])
            lesion_slice_sitk.SetDirection(direction_2d)

            # Construct the new file name for LESION
            lesion_file = os.path.join(lesion_dir, f'LESION_{current_index}.nii')
            sitk.WriteImage(lesion_slice_sitk, lesion_file)
            print(f'Saved {lesion_file}')

            current_index += 1

    return current_index

def organize_and_extract_slices(input_dir, output_dir, patient_indices):
    """Organizes the patient data and extracts axial slices."""
    # Define the FLAIR and LESION directories within the output directory
    flair_dir = os.path.join(output_dir, 'FLAIR')
    lesion_dir = os.path.join(output_dir, 'LESION')

    # Create the output directories if they don't exist
    os.makedirs(flair_dir, exist_ok=True)
    os.makedirs(lesion_dir, exist_ok=True)

    current_index = 0

    # Iterate through each specified patient index
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
                current_index = get_axial_slices(flair_image, lesion_image, flair_dir, lesion_dir, current_index)
        else:
            print(f'Patient folder {patient_folder} does not exist')


if __name__ == '__main__':
    organize_and_extract_slices("data", "org_data_2", accepted_indexes)
