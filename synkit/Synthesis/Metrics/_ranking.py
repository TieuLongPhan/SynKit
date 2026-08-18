import statistics
from typing import List, Dict


def _coverage(
    reactions_data: List[Dict[str, str]], key_ground_truth: str, key_prediction: str
) -> float:
    """Calculates the coverage percentage, which measures how many of the
    predicted reactions exactly match the ground truth reactions given in a
    list of dictionaries.

    :param reactions_data: List of dictionaries containing
                           reaction SMILES strings.
    :type reactions_data: List[Dict[str, str]]
    :param key_ground_truth: Key in the dictionary for the ground truth reaction SMILES.
    :type key_ground_truth: str
    :param key_prediction: Key in the dictionary for the predicted reaction SMILES.
    :type key_prediction: str

    :return: The coverage percentage.
    :rtype: float
    """
    correct_matches = sum(
        1
        for reaction in reactions_data
        if reaction.get(key_ground_truth) in reaction.get(key_prediction)
    )
    return (correct_matches / len(reactions_data)) * 100 if reactions_data else 0


def _novelty_rate(
    reactions_data: List[Dict[str, any]], key_ground_truth: str, key_prediction: str
) -> float:
    """Calculates the False Positive Rate (FPR) for each observation and then
    averages these values across all observations. The FPR represents the
    proportion of predictions that do not match the ground truth for each
    individual entry in the dataset.

    :param reactions_data: List of dictionaries containing
                           the ground truth and predicted reactions, where predictions are given as
                           a list of RSMIs.
    :type reactions_data: List[Dict[str, any]]
    :param key_ground_truth: Dictionary key to access the ground truth reaction RSMI.
    :type key_ground_truth: str
    :param key_prediction: Dictionary key to access the list of predicted reaction RSMIs.
    :type key_prediction: str

    :return: Average false-positive rate as a percentage.
    :rtype: float
    """
    fpr_list = []  # List to store FPR for each observation

    for entry in reactions_data:
        ground_truth = entry.get(key_ground_truth)
        predictions = entry.get(key_prediction, [])

        if not predictions:  # Skip if no predictions
            continue

        # Calculate FPR for this entry
        false_positives = sum(1 for pred in predictions if pred != ground_truth)
        total_predictions = len(predictions)
        fpr_list.append((false_positives / total_predictions) * 100)

    # Calculate and return the mean FPR
    return statistics.mean(fpr_list) if fpr_list else 0.0


def _recognition_rate(
    reactions_data: List[Dict[str, any]], key_ground_truth: str, key_prediction: str
) -> float:
    """Calculates the recognition rate for each observation and averages these
    rates across all observations. The recognition rate measures the proportion
    of the prediction list that matches the single ground truth reaction for
    each entry.

    :param reactions_data: List of dictionaries containing
                           the ground truth
                           and predicted reactions, where the ground truth is a single RSMI and predictions
                           are lists of RSMIs.
    :type reactions_data: List[Dict[str, any]]
    :param key_ground_truth: Dictionary key to access the ground truth reaction RSMIs.
    :type key_ground_truth: str
    :param key_prediction: Dictionary key to access the list of predicted reaction RSMIs.
    :type key_prediction: str

    :return: Average recognition rate as a percentage.
    :rtype: float
    """
    recognition_rates = []

    for entry in reactions_data:
        ground_truth = entry.get(key_ground_truth)
        predictions = entry.get(key_prediction, [])

        if predictions:

            matches = sum(1 for pred in predictions if pred == ground_truth)

            recognition_rate = (matches / len(predictions)) * 100
            recognition_rates.append(recognition_rate)
        else:
            recognition_rates.append(0.0)

    return statistics.mean(recognition_rates) if recognition_rates else 0.0


def _top_k_accuracy(
    reactions_data: List[Dict[str, any]],
    key_ground_truth: str,
    key_prediction: str,
    k: int,
) -> float:
    """Calculates the Top-K accuracy by using the coverage function on the top
    K predictions. This measures the probability that the true reaction is
    within the top K predictions.

    :param reactions_data: List of dictionaries containing
                           RSMI strings.
    :type reactions_data: List[Dict[str, any]]
    :param key_ground_truth: Key in the dictionary for the ground truth RSMI.
    :type key_ground_truth: str
    :param key_prediction: Key in the dictionary for the predicted RSMIs
                           (list of predictions).
    :type key_prediction: str
    :param k: The number of top predictions to consider.
    :type k: int

    :return: The Top-K accuracy percentage.
    :rtype: float
    """
    modified_data = [
        {**entry, "Top_K_Predictions": entry[key_prediction][:k]}
        for entry in reactions_data
    ]
    return _coverage(modified_data, key_ground_truth, "Top_K_Predictions")


def _calculate_f_beta_score(
    recognition_rate: float,  # This serves as the precision
    coverage_rate: float,  # This serves as the recall
    beta: float = 1.0,  # Beta factor, default is 1.0 for F1 score
) -> float:
    """Computes the F-beta Score, which is a weighted harmonic mean of
    recognition rate and coverage rate. The recognition rate (precision) and
    coverage rate (recall) must be expressed as percentages. A beta value of
    1.0 means equal importance to precision and recall (F1 Score), greater than
    1.0 gives more importance to recall (e.g., F2 Score), and less than 1.0
    prioritizes precision (e.g., F0.5 Score).

    :param recognition_rate: The recognition rate of the predictions,
                             acting as precision, expected to be between 0 and 100.
    :type recognition_rate: float
    :param coverage_rate: The coverage rate of the predictions, acting as recall,
                          expected to be between 0 and 100.
    :type coverage_rate: float
    :param beta: The weight emphasizing recall over precision. Default is 1.0.
    :type beta: float

    :return: The F-beta Score as a percentage, which balances precision and recall based on the beta factor.
    :rtype: float
    """
    if recognition_rate == 0 or coverage_rate == 0:
        return 0  # If either rate is zero, F-beta is zero to avoid division by zero

    # Calculate precision and recall
    precision = recognition_rate / 100
    recall = coverage_rate / 100

    # Calculate F-beta Score using the formula:
    # F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    beta_squared = beta**2
    f_beta_score = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return f_beta_score
