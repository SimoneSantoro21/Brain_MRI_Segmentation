import pytest
import numpy as np

from libs.evaluation_metrics import precision_score
from libs.evaluation_metrics import recall_score
from libs.evaluation_metrics import accuracy
from libs.evaluation_metrics import jaccard_index
from libs.evaluation_metrics import dice_coeff


def test_precision_score():
    """
    Testing the precision score output for a given pair of tensors.

    GIVEN: Two different tensors, ground truth and prediction mask
    WHEN: Precision score is computed between the two
    THEN: The result value is equal to the expected one and is a scalar Float
    """
    ground_truth = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    pred_mask = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]])
    
    result = precision_score(ground_truth, pred_mask)

    assert result == 0.8
    assert isinstance(result, float)
    assert np.isscalar(result)


def test_recall_score():
    """
    Testing the recall score output for a given pair of tensors.

    GIVEN: Two different tensors, ground truth and prediction mask
    WHEN: Recall score is computed between the two
    THEN: The result value is equal to the expected one and is a scalar Float
    """
    ground_truth = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    pred_mask = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]])
    
    result = recall_score(ground_truth, pred_mask)

    assert result == 0.8
    assert isinstance(result, float)
    assert np.isscalar(result)


def test_accuracy():
    """
    Testing the accuracy score output for a given pair of tensors.

    GIVEN: Two different tensors, ground truth and prediction mask
    WHEN: Accuracy score is computed between the two
    THEN: The result value is equal to the expected one and is a scalar Float
    """
    ground_truth = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    pred_mask = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]])
    
    result = accuracy(ground_truth, pred_mask)
    assert result == 0.778
    assert isinstance(result, float)
    assert np.isscalar(result)


def test_jaccard():
    """
    Testing the Jaccard Index output for a given pair of tensors.

    GIVEN: Two different tensors, ground truth and prediction mask
    WHEN: Jaccard Index is computed between the two
    THEN: The result value is equal to the expected one and is a scalar Float
    """
    ground_truth = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    pred_mask = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]])
    
    result = jaccard_index(ground_truth, pred_mask)

    assert result == 0.667
    assert isinstance(result, float)
    assert np.isscalar(result)


def test_dice():
    """
    Testing the dice coefficient output for a given pair of tensors.

    GIVEN: Two different tensors, ground truth and prediction mask
    WHEN: Dice coefficient is computed between the two
    THEN: The result value is equal to the expected one and is a scalar Float
    """
    ground_truth = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    pred_mask = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]])
    
    result = dice_coeff(ground_truth, pred_mask)

    assert result == 0.8
    assert isinstance(result, float)
    assert np.isscalar(result)


def test_identical_tensors():
    """
    Testing that the metrics values computed on two identical tensors are equal to 1

    GIVEN: A pair of two identical tensors
    WHEN: Metrics are computed between the two
    THEN: The output value for all the metrics is 1
    """
    ground_truth = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    pred_mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    
    assert precision_score(ground_truth, pred_mask) == 1.0
    assert recall_score(ground_truth, pred_mask) == 1.0
    assert accuracy(ground_truth, pred_mask) == 1.0
    assert jaccard_index(ground_truth, pred_mask) == 1.0
    assert dice_coeff(ground_truth, pred_mask) == 1.0


def test_totally_different_tensors():
    """
    Testing that the metrics values computed on two completely different tensors are equal to 0

    GIVEN: A pair of two completely different tensors
    WHEN: Metrics are computed between the two
    THEN: The output value for all the metrics is 0
    """
    ground_truth = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    pred_mask = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    assert precision_score(ground_truth, pred_mask) == 0.0
    assert recall_score(ground_truth, pred_mask) == 0.0
    assert accuracy(ground_truth, pred_mask) == 0.0
    assert jaccard_index(ground_truth, pred_mask) == 0.0
    assert dice_coeff(ground_truth, pred_mask) == 0.0