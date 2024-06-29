import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import os

from libs.U_Net import UNet
from libs.pre_processing_functions import pre_processing


def prediction(image_path, mask_path, model_path, device):
    """
    Predicts the segmentation map from an image.

    Args:
        - image_path: path of the image to segment
        - mask_path: path of the ground truth segmentation mask
        - device: CPU or GPU

    Returns:
        - predicted segmentation mask
    """
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location= torch.device(device)))
    model.to(device)

    image_raw = sitk.ReadImage(image_path)
    image = pre_processing(image_raw)

    mask_sitk = sitk.ReadImage(mask_path)
    mask_tensor = pre_processing(mask_sitk)

    image = image.to(device).unsqueeze(0)
    mask_tensor = mask_tensor.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        for i in range(50):
            prediction = model(image)
        
    prediction = prediction.to("cpu")

    image = image.squeeze(0).cpu().detach()
    image.permute(1, 2, 0)

    prediction = prediction.squeeze(0).cpu().detach()    
    prediction.permute(1, 2, 0)
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1

    image_np = image.squeeze().numpy()
    prediction_np = prediction.squeeze().numpy()
    mask_np = mask_tensor.squeeze().cpu().detach().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image_np, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(prediction_np, cmap='gray')
    ax[1].set_title('Prediction')

    ax[2].imshow(mask_np, cmap='gray')
    ax[2].set_title('Ground Truth Mask')
    plt.show()

    print(prediction_np)


if __name__ == "__main__":
    DATA_PATH = "org_data"
    FLAIR_PATH = os.path.join(DATA_PATH, "FLAIR")
    LESION_PATH = os.path.join(DATA_PATH, "LESION")

    #MODEL_SAVE_PATH = "C:/Users/User/Brain_MRI_Segmentation/model/unet_1e-4_a0.25_g2.pth"
    MODEL_SAVE_PATH = "C:/Users/User/Brain_MRI_Segmentation/model/unet.pth"

    IMAGE_PATH = os.path.join(FLAIR_PATH, "FLAIR_200.nii")
    MASK_PATH = os.path.join(LESION_PATH, "LESION_200.nii")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prediction(IMAGE_PATH, MASK_PATH, MODEL_SAVE_PATH, device)


    