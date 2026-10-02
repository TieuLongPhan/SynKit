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
    """Computes the metrics for a list of reactions data.

    :param reactions_data: List of dictionaries containing RSMI strings.
    :type reactions_data: List[Dict[str, any]]
    :param key_ground_truth: Key in the dictionary for the ground truth RSMI.
    :type key_ground_truth: str
    :param key_prediction: Key in the dictionary for the predicted RSMIs
                           (list of predictions).
    :type key_prediction: str
    :param k: The number of top predictions to consider. Defaults to 5.
    :type k: int, optional
    :param beta: Beta parameter for the F-beta score. Defaults to 1.
    :type beta: float, optional

    :return: A dictionary with the metrics.
    :rtype: Dict[str, float]
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
