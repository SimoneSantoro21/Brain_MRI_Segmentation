import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import os
import argparse

from libs.U_Net import UNet
from libs.pre_processing_functions import pre_processing

def arg_parser():
    description = 'Bayesian U-Net prediction'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--input',
                        required=True,
                        type=str,
                        help='Path of the input data directory')
    
    parser.add_argument('--index',
                        required=True,
                        type=int,
                        help='Slice index in dataset directory')
    
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        help='Model to be used for inference')
    
    parser.add_argument('--threshold',
                        default=0.5,
                        type=float,
                        help='Threshold value for output binarization')
    
    parser.add_argument('--plot',
                        default=True,
                        type=bool,
                        help='Choose to visualize or not the output')
    
    parser.add_argument('--forward',
                        default=50,
                        type=int,
                        help='Number of stochastic forward passes of the model')
    
    args = parser.parse_args()
    return args


def prediction(image_path, mask_path, model_path, index, threshold_value=0.5, plot=True, num_samples=50):
    """
    Predicts the segmentation map from an image using Bayesian U-Net.

    Args:
        - image_path (str): path of the image to segment
        - mask_path (str): path of the ground truth segmentation mask
        - model_path (str): path of the model checkpoint
        - index (int): index of the image, needed for naming output dir
        - threshold_value (float): determines the threshold value to binarize model prediction
        - plot (bool): Determines whether to plot the prediction or not
        - num_samples (int): number of stochastic forward passes

    Returns:
        - mean_prediction: mean predicted segmentation mask
        - variance_prediction: variance of predicted segmentation masks
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.train()  # Needed to keep dropout layers

    image_raw = sitk.ReadImage(image_path)
    image = pre_processing(image_raw)

    mask_sitk = sitk.ReadImage(mask_path)
    mask_tensor = pre_processing(mask_sitk)

    image = image.to(device).unsqueeze(0)
    mask_tensor = mask_tensor.to(device)

    # Inference
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            prediction = model(image)
            prediction = prediction.to("cpu").squeeze(0).squeeze(0).detach().numpy()
            predictions.append(prediction)

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    mean_prediction[mean_prediction < threshold_value] = 0
    mean_prediction[mean_prediction >= threshold_value] = 1
    variance_prediction = np.var(predictions, axis=0)

    # Saving mean and variance output 
    subdirectory = os.path.join("predictions", f"Prediction_index-{index}")
    os.makedirs(subdirectory, exist_ok=True)
    plt.imsave(os.path.join(subdirectory, "mean_prediction.png"), mean_prediction, cmap='grey')
    plt.imsave(os.path.join(subdirectory, "variance_prediction.png"), variance_prediction, cmap='hot')

    if plot:
        image_np = image.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
        mask_np = mask_tensor.squeeze().cpu().detach().numpy()

        plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        ax[0].imshow(image_np, cmap='gray')
        ax[0].imshow(mask_np, cmap='hot', alpha=0.3)
        ax[0].set_title('Original Image & gr. truth', fontsize=25)
        ax[0].tick_params(axis='both', which='major', labelsize=12)

        ax[1].imshow(image_np, cmap='gray')
        ax[1].imshow(mean_prediction, cmap='hot', alpha=0.3)  # Overlay mean prediction
        ax[1].set_title('Mean Prediction', fontsize=25)
        ax[1].tick_params(axis='both', which='major', labelsize=12)

        ax[2].imshow(image_np, cmap='gray')
        variance_overlay = ax[2].imshow(variance_prediction, cmap='hot', alpha=0.5)  # Overlay variance
        ax[2].set_title('Variance Prediction', fontsize=25)
        cbar = fig.colorbar(variance_overlay, ax=ax[2], fraction=0.046, pad=0.04)
        ax[2].tick_params(axis='both', which='major', labelsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(subdirectory, "comparison.png"))
        plt.show()

    return mean_prediction, variance_prediction


if __name__ == "__main__":
    args = arg_parser()

    DATA_PATH = args.input
    FLAIR_PATH = os.path.join(DATA_PATH, "FLAIR")
    LESION_PATH = os.path.join(DATA_PATH, "LESION_1")

    IMAGE_PATH = os.path.join(FLAIR_PATH, f"FLAIR_{args.index}.nii")
    MASK_PATH = os.path.join(LESION_PATH, f"LESION_{args.index}.nii")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = args.model
    threshold_value = args.threshold
    plot = args.plot
    num_samples = args.forward
    index = args.index

    bayesian_prediction = prediction(IMAGE_PATH, MASK_PATH, model, index, threshold_value, plot, num_samples)
    


    