import tensorflow as tf
import numpy as np
from squad_albert.metrics import get_start_end_arrays


def CE_loss(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Description: Computes the combined Categorical Cross-Entropy loss for start and end index predictions.
    Parameters:
    ground_truth (np.ndarray): Ground truth tensor of shape [B, max_sequence_length, 2].
    prediction (np.ndarray): Predicted tensor of shape [B, max_sequence_length, 2].
    Returns:
    combined_loss (np.ndarray): Combined loss for start and end index predictions, computed as the sum of individual Categorical Cross-Entropy losses weighted by alpha.
    """
    ground_truth = tf.convert_to_tensor(ground_truth)
    prediction = tf.convert_to_tensor(prediction)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    alpha = 1.0
    start_pred, end_pred = get_start_end_arrays(prediction)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    combined_loss = loss(start_gt, start_pred) + alpha * loss(end_gt, end_pred)
    return combined_loss.numpy()