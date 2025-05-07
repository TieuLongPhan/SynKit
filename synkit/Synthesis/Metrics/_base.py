from typing import List, Dict
from ._ranking import (
    _novelty_rate,
    _coverage,
    _recognition_rate,
    _top_k_accuracy,
    _calculate_f_beta_score,
)


def _compute_metrics(
    reactions_data: List[Dict[str, any]],
    key_ground_truth: str,
    key_prediction: str,
    k: int = 5,
    beta: float = 1,
) -> Dict[str, float]:
    """
    Computes the metrics for a list of reactions data.

    Parameters:
    - reactions_data (List[Dict[str, any]]): List of dictionaries containing RSMI strings.
    - key_ground_truth (str): Key in the dictionary for the ground truth RSMI.
    - key_prediction (str): Key in the dictionary for the predicted RSMIs
    (list of predictions).
    - k (int, optional): The number of top predictions to consider. Defaults to 5.
    - alpha (float, optional): Weight for the novelty component. Defaults to 0.5.
    - beta (float, optional): Weight for the recognition component. Defaults to 0.5.

    Returns:
    - Dict[str, float]: A dictionary with the metrics.
    """
    return {
        "Novelty": _novelty_rate(reactions_data, key_ground_truth, key_prediction)
        / 100,
        "Coverage": _coverage(reactions_data, key_ground_truth, key_prediction) / 100,
        "Recognition": _recognition_rate(
            reactions_data, key_ground_truth, key_prediction
        )
        / 100,
        f"Top_{k}_Accuracy": _top_k_accuracy(
            reactions_data, key_ground_truth, key_prediction, k
        )
        / 100,
        f"F{beta}_score": _calculate_f_beta_score(
            _recognition_rate(reactions_data, key_ground_truth, key_prediction),
            _coverage(reactions_data, key_ground_truth, key_prediction),
            beta,
        ),
    }
