import SimpleITK as sitk
import os
import argparse
import csv

from libs.pre_processing_functions import pre_processing
from libs.evaluation_metrics import precision_score
from libs.evaluation_metrics import recall_score
from libs.evaluation_metrics import accuracy
from libs.evaluation_metrics import jaccard_index
from libs.evaluation_metrics import dice_coeff


def arg_parser():
    description = 'Bayesian U-Net evaluation'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--index',
                        required=True,
                        type=int,
                        help='Slice index in test dataset directory')
    
    args = parser.parse_args()
    return args


def evaluate(image_path, gr_truth_path):
    """
    Evaluates the segmentation performance of a predicted image against a ground truth mask.

    Args:
        image_path (str): Path to the predicted segmentation image file.
        gr_truth_path (str): Path to the ground truth segmentation mask file.

    Returns:
        dict: A dictionary containing the following segmentation metrics:
            - "precision": Precision score.
            - "recall": Recall score.
            - "acc": Accuracy.
            - "jaccard": Jaccard index (Intersection over Union).
            - "dice": Dice coefficient.
    """
    image_sitk = sitk.ReadImage(image_path)
    image_tensor = pre_processing(image_sitk)
    image_np = image_tensor.squeeze().cpu().detach().numpy()

    mask_sitk = sitk.ReadImage(gr_truth_path)
    mask_tensor = pre_processing(mask_sitk)
    mask_np = mask_tensor.squeeze().cpu().detach().numpy()

    evaluated_metrics = {}
    evaluated_metrics["precision"] = precision_score(mask_np, image_np)
    evaluated_metrics["recall"] = recall_score(mask_np, image_np)
    evaluated_metrics["acc"] = accuracy(mask_np, image_np)
    evaluated_metrics["jaccard"] = jaccard_index(mask_np, image_np)
    evaluated_metrics["dice"] = dice_coeff(mask_np, image_np)

    return evaluated_metrics


def save_metrics_to_csv(metrics, csv_path):
    """
    Saves the evaluation metrics to a CSV file.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics.
        csv_path (str): Path to the CSV file where the metrics will be saved.
    
        Returns:
            None
    """
    csv_file = os.path.join(csv_path, "evaluation_metrics.csv")
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())
    print(f"Metrics saved to {csv_file}")

    return None


if __name__ == "__main__":
    args = arg_parser()
    index = args.index

    DATA_PATH = "dataset/testing"
    PRED_PATH = f"predictions/Prediction_index-{index}"
    LESION_PATH = os.path.join(DATA_PATH, "LESION")

    IMAGE_PATH = os.path.join(PRED_PATH, "mean_prediction.png")
    MASK_PATH = os.path.join(LESION_PATH, f"LESION_{index}.nii")

    metrics1 = evaluate(IMAGE_PATH, MASK_PATH)

    print("METRICS FOR PREDICTION VS GROUND TRUTH")
    print("| Metric            | Value      |")  
    print("|-------------------|-------------|")
    print(f"| Precision score   | {metrics1['precision']:.3f}|")
    print(f"| Recall score      | {metrics1['recall']:.3f}|")
    print(f"| Accuracy          | {metrics1['acc']:.3f}|")
    print(f"| Jaccard Index     | {metrics1['jaccard']:.3f}|")
    print(f"| Dice coefficient  | {metrics1['dice']:.3f}|")

    save_metrics_to_csv(metrics1, PRED_PATH)









