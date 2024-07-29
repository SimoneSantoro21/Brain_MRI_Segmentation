import numpy as np


def precision_score(ground_truth, pred_mask):
    """
    Compute the precision score between the ground truth and 
    the predicted segmentation mask.

    Args:
        -ground_truth: np.ndarray reperesenting the 2D binary ground truth image
        -pred_mask: np.ndarray reperesenting the 2D prediction of the model

    returns:
        -precision: Value of the prediction score, rounded to the third decimal
    """
    intersect = np.sum(pred_mask*ground_truth)
    total_pixel_pred = np.sum(pred_mask)
    if total_pixel_pred == 0:
        precision = 0
    else:
        precision = np.mean(intersect/total_pixel_pred)

    return round(precision, 3)


def recall_score(ground_truth, pred_mask):
    """
    Compute the recall score between the ground truth and 
    the predicted segmentation mask.

    Args:
        -ground_truth: np.ndarray reperesenting the 2D binary ground truth image
        -pred_mask: np.ndarray reperesenting the 2D prediction of the model

    returns:
        -recall: Value of the recall score, rounded to the third decimal
    """
    intersect = np.sum(pred_mask*ground_truth)
    total_pixel_truth = np.sum(ground_truth)
    if total_pixel_truth == 0:
        recall = 0
    else:
        recall = np.mean(intersect/total_pixel_truth)

    return round(recall, 3)


def accuracy(ground_truth, pred_mask):
    """
    Compute the accuracy score between the ground truth and 
    the predicted segmentation mask.

    Args:
        -ground_truth: np.ndarray reperesenting the 2D binary ground truth image
        -pred_mask: np.ndarray reperesenting the 2D prediction of the model

    returns:
        -acc: Value of the accuracy score, rounded to the third decimal
    """
    intersect = np.sum(pred_mask*ground_truth)
    union = np.sum(pred_mask) + np.sum(ground_truth) - intersect
    xor = np.sum(ground_truth==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))

    return round(acc, 3)


def jaccard_index(ground_truth, pred_mask):
    """
    Compute the Jaccardi Index(IoU) between the ground truth and 
    the predicted segmentation mask.

    Args:
        -ground_truth: np.ndarray reperesenting the 2D binary ground truth image
        -pred_mask: np.ndarray reperesenting the 2D prediction of the model

    returns:
        -jaccard: Value of the Jaccardi Index, rounded to the third decimal
    """
    intersect = np.sum(pred_mask*ground_truth)
    union = np.sum(pred_mask) + np.sum(ground_truth) - intersect
    if union == 0:
        jaccard_coeff = 0
    else:
        jaccard_coeff = np.mean(intersect/union)

    return round(jaccard_coeff, 3)


def dice_coeff(ground_truth, pred_mask):
    """
    Compute the Dice Coefficient between the ground truth and 
    the predicted segmentation mask.

    Args:
        -ground_truth: np.ndarray reperesenting the 2D binary ground truth image
        -pred_mask: np.ndarray reperesenting the 2D prediction of the model

    returns:
        -dice_coeff: Value of the Dice Coefficient, rounded to the third decimal
    """
    intersect = np.sum(pred_mask*ground_truth)
    union = np.sum(pred_mask) + np.sum(ground_truth)
    if union == 0:
        dice_coeff = 0
    else:
        dice_coeff = (2. * intersect) / union

    return round(dice_coeff, 3)
