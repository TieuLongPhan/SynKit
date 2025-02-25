import statistics
from typing import List, Dict


def _coverage(
    reactions_data: List[Dict[str, str]], key_ground_truth: str, key_prediction: str
) -> float:
    """
    Calculates the coverage percentage, which measures how many of the predicted reactions
    exactly match the ground truth reactions given in a list of dictionaries.

    Parameters:
    - reactions_data (List[Dict[str, str]]): List of dictionaries containing
    reaction SMILES strings.
    - key_ground_truth (str): Key in the dictionary for the ground truth reaction SMILES.
    - key_prediction (str): Key in the dictionary for the predicted reaction SMILES.

    Returns:
    - float: The coverage percentage.
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
    """
    Calculates the False Positive Rate (FPR) for each observation and then averages
    these values across all observations. The FPR represents the proportion of
    predictions that do not match
    the ground truth for each individual entry in the dataset.

    Parameters:
    - reactions_data (List[Dict[str, any]]): List of dictionaries containing
    the ground truth and predicted reactions, where predictions are given as
    a list of RSMIs.
    - key_ground_truth (str): Dictionary key to access the ground truth reaction RSMI.
    - key_prediction (str): Dictionary key to access the list of predicted reaction RSMIs.

    Returns:
    - float: The average False Positive Rate (FPR) as a percentage,
    indicating the average ratio of incorrect predictions to
    total predictions across all observations.
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
    """
    Calculates the recognition rate for each observation and averages these rates
    across all observations. The recognition rate measures the proportion of
    the prediction list that matches the single ground truth reaction for each entry.

    Parameters:
    - reactions_data (List[Dict[str, any]]): List of dictionaries containing
    the ground truth
      and predicted reactions, where the ground truth is a single RSMI and predictions
      are lists of RSMIs.
    - key_ground_truth (str): Dictionary key to access the ground truth reaction RSMIs.
    - key_prediction (str): Dictionary key to access the list of predicted reaction RSMIs.

    Returns:
    - float: The average recognition rate as a percentage, indicating the average
    proportion of correct predictions relative to the total number of predictions
    across all observations.
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
    """
    Calculates the Top-K accuracy by using the coverage function on the top K predictions.
    This measures the probability that the true reaction is within the top K predictions.

    Parameters:
    - reactions_data (List[Dict[str, any]]): List of dictionaries containing
    RSMI strings.
    - key_ground_truth (str): Key in the dictionary for the ground truth RSMI.
    - key_prediction (str): Key in the dictionary for the predicted RSMIs
    (list of predictions).
    - k (int): The number of top predictions to consider.

    Returns:
    - float: The Top-K accuracy percentage.
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
    """
    Computes the F-beta Score, which is a weighted harmonic mean of recognition rate
    and coverage rate. The recognition rate (precision) and coverage rate (recall)
    must be expressed as percentages. A beta value of 1.0 means equal importance to
    precision and recall (F1 Score), greater than 1.0 gives more importance to recall
    (e.g., F2 Score), and less than 1.0 prioritizes precision (e.g., F0.5 Score).

    Parameters:
    - recognition_rate (float): The recognition rate of the predictions,
    acting as precision, expected to be between 0 and 100.
    - coverage_rate (float): The coverage rate of the predictions, acting as recall,
    expected to be between 0 and 100.
    - beta (float): The weight emphasizing recall over precision. Default is 1.0.

    Returns:
    - float: The F-beta Score as a percentage, which balances precision and recall
    based on the beta factor.
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
