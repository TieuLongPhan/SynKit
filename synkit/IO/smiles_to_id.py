import time
import requests
import urllib.parse
from typing import List
from joblib import Parallel, delayed


def smiles_to_iupac(smiles_string: str, timeout: int = 1):
    """
    Converts a SMILES string to its corresponding IUPAC name(s) using the PubChem PUG REST API.

    Parameters:
    - smiles_string (str): The SMILES string of the compound (e.g., "C=O" for formaldehyde).
    - timeout (int, optional): The timeout in seconds for the request. Default is 1 second.

    Returns:
    - list: A list of IUPAC names associated with the SMILES string. Returns an empty list if none found.
    """
    # URL encode the SMILES string to handle special characters
    encoded_smiles = urllib.parse.quote(smiles_string)

    # PubChem PUG REST API endpoint to retrieve properties
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/property/IUPACName/JSON"

    retries = 3  # Number of retries in case of failure
    delay = 2  # Delay between retries (in seconds)

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)  # Adjust timeout for speed
            response.raise_for_status()  # Raise an HTTPError for bad responses

            data = response.json()

            # Extract the IUPAC name(s) from the response
            properties = data.get("PropertyTable", {}).get("Properties", [])

            if not properties:
                print(f"No properties found for SMILES: {smiles_string}")
                return []

            iupac_names = [
                prop.get("IUPACName") for prop in properties if prop.get("IUPACName")
            ]

            if iupac_names:
                return iupac_names
            else:
                print(f"No IUPAC name found for SMILES: {smiles_string}")
                return []

        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            # If an error occurs, retry a few times
            print(
                f"Attempt {attempt + 1} failed for SMILES: {smiles_string}, Error: {e}"
            )
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                print(f"Final failure for SMILES: {smiles_string}")
                return []

    return []


def batch_process_smiles(smiles_batch: List[str], timeout=1):
    """
    Processes a batch of SMILES strings to get IUPAC names.

    Parameters:
    - smiles_batch (list): A list of SMILES strings to process.
    - timeout (int): Timeout for requests (in seconds).

    Returns:
    - list: A list of IUPAC name results for each SMILES in the batch.
    """
    return [smiles_to_iupac(smiles, timeout) for smiles in smiles_batch]


def get_iupac_for_smiles_list(
    smiles_list: List[str], batch_size=10, n_jobs=4, timeout=1
):
    """
    Convert a list of SMILES strings to their corresponding IUPAC names using the PubChem API with batch processing.

    Parameters:
    smiles_list (list): A list of SMILES strings to be converted to IUPAC names.
    batch_size (int): Number of SMILES strings to process in each batch.
    n_jobs (int): Number of parallel jobs to run for batch processing.
    timeout (int): Timeout for requests (in seconds).

    Returns:
    dict: A dictionary with SMILES as keys and lists of IUPAC names as values.
    """
    # Split the list into smaller batches
    # fmt: off
    batches = [
        smiles_list[i: i + batch_size] for i in range(0, len(smiles_list), batch_size)
    ]
    # fmt: on

    # Use joblib's Parallel and delayed to process batches in parallel
    batch_results = Parallel(n_jobs=n_jobs)(
        delayed(batch_process_smiles)(batch, timeout) for batch in batches
    )

    # Flatten the list of results and map to SMILES
    flattened_results = [item for sublist in batch_results for item in sublist]
    iupac_dict = dict(zip(smiles_list, flattened_results))

    return iupac_dict


# Example of usage
smiles_list = ["CCO", "C=O", "CC(=O)O", "C1=CC=CC=C1", "C2H6O", "C4H10", "C5H12"]
iupac_results = get_iupac_for_smiles_list(smiles_list, batch_size=3, n_jobs=2)

for smiles, iupac_names in iupac_results.items():
    print(f"SMILES: {smiles} => IUPAC Names: {iupac_names}")
