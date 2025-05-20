import os
import json
import pickle
import numpy as np
from numpy import ndarray
from joblib import dump, load
from typing import List, Dict, Any, Generator, Optional
from synkit.IO.debug import setup_logging

logger = setup_logging()


def save_database(database: List[Dict], pathname: str = "./Data/database.json") -> None:
    """
    Save a database (a list of dictionaries) to a JSON file.

    :param database: The database to be saved.
    :type database: list[dict]
    :param pathname: The path where the database will be saved. Defaults to './Data/database.json'.
    :type pathname: str

    :raises TypeError: If the database is not a list of dictionaries.
    :raises ValueError: If there is an error writing the file.
    """
    if not all(isinstance(item, dict) for item in database):
        raise TypeError("Database should be a list of dictionaries.")
    try:
        with open(pathname, "w") as f:
            json.dump(database, f)
    except IOError as e:
        raise ValueError(f"Error writing to file {pathname}: {e}")


def load_database(pathname: str = "./Data/database.json") -> List[Dict]:
    """
    Load a database (a list of dictionaries) from a JSON file.

    :param pathname: The path from where the database will be loaded. Defaults to './Data/database.json'.
    :type pathname: str

    :returns: The loaded database.
    :rtype: list[dict]

    :raises ValueError: If there is an error reading the file.
    """
    try:
        with open(pathname, "r") as f:
            database = json.load(f)
        return database
    except IOError as e:
        raise ValueError(f"Error reading file {pathname}: {e}")


def save_to_pickle(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Save a list of dictionaries to a pickle file.

    :param data: A list of dictionaries to be saved.
    :type data: list[dict]
    :param filename: The name of the file where the data will be saved.
    :type filename: str
    """
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_from_pickle(filename: str) -> List[Any]:
    """
    Load data from a pickle file.

    :param filename: The name of the pickle file to load data from.
    :type filename: str

    :returns: The data loaded from the pickle file.
    :rtype: list
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def load_gml_as_text(gml_file_path: str) -> Optional[str]:
    """
    Load the contents of a GML file as a text string.

    :param gml_file_path: The file path to the GML file.
    :type gml_file_path: str

    :returns: The text content of the GML file, or None if the file does not exist or an error occurs.
    :rtype: str or None
    """
    try:
        with open(gml_file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"File not found: {gml_file_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while reading {gml_file_path}: {e}")
        return None


def save_text_as_gml(gml_text: str, file_path: str) -> bool:
    """
    Save a GML text string to a file.

    :param gml_text: The GML content as a text string.
    :type gml_text: str
    :param file_path: The file path where the GML text will be saved.
    :type file_path: str

    :returns: True if saving was successful, False otherwise.
    :rtype: bool
    """
    try:
        with open(file_path, "w") as file:
            file.write(gml_text)
        logger.info(f"GML text successfully saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"An error occurred while saving the GML text: {e}")
        return False


def save_compressed(array: ndarray, filename: str) -> None:
    """
    Saves a NumPy array in a compressed format using .npz extension.

    :param array: The NumPy array to be saved.
    :type array: numpy.ndarray
    :param filename: The file path or name to save the array to, with a '.npz' extension.
    :type filename: str
    """
    np.savez_compressed(filename, array=array)


def load_compressed(filename: str) -> ndarray:
    """
    Loads a NumPy array from a compressed .npz file.

    :param filename: The path of the .npz file to load.
    :type filename: str

    :returns: The loaded NumPy array.
    :rtype: numpy.ndarray

    :raises KeyError: If the .npz file does not contain an array with the key 'array'.
    """
    with np.load(filename) as data:
        if "array" in data:
            return data["array"]
        else:
            raise KeyError(
                "The .npz file does not contain an array with the key 'array'."
            )


def save_model(model: Any, filename: str) -> None:
    """
    Save a machine learning model to a file using joblib.

    :param model: The machine learning model to save.
    :type model: object
    :param filename: The path to the file where the model will be saved.
    :type filename: str
    """
    dump(model, filename)
    logger.info(f"Model saved successfully to {filename}")


def load_model(filename: str) -> Any:
    """
    Load a machine learning model from a file using joblib.

    :param filename: The path to the file from which the model will be loaded.
    :type filename: str

    :returns: The loaded machine learning model.
    :rtype: object
    """
    model = load(filename)
    logger.info(f"Model loaded successfully from {filename}")
    return model


def save_dict_to_json(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a JSON file.

    :param data: The dictionary to be saved.
    :type data: dict
    :param file_path: The path to the file where the dictionary should be saved.
    :type file_path: str
    """
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"Dictionary successfully saved to {file_path}")


def load_dict_from_json(file_path: str) -> Optional[dict]:
    """
    Load a dictionary from a JSON file.

    :param file_path: The path to the JSON file from which to load the dictionary.
    :type file_path: str

    :returns: The dictionary loaded from the JSON file, or None if an error occurs.
    :rtype: dict or None
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        logger.info(f"Dictionary successfully loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(e)
        return None


def load_from_pickle_generator(file_path: str) -> Generator[Any, None, None]:
    """
    A generator that yields items from a pickle file where each pickle load returns a list of dictionaries.

    :param file_path: The path to the pickle file to load.
    :type file_path: str

    :yields: A single item from the list of dictionaries stored in the pickle file.
    :rtype: Any
    """
    with open(file_path, "rb") as file:
        while True:
            try:
                batch_items = pickle.load(file)
                for item in batch_items:
                    yield item
            except EOFError:
                break


def collect_data(num_batches: int, temp_dir: str, file_template: str) -> List[Any]:
    """
    Collects and aggregates data from multiple pickle files into a single list.

    :param num_batches: The number of batch files to process.
    :type num_batches: int
    :param temp_dir: The directory where the batch files are stored.
    :type temp_dir: str
    :param file_template: The template string for batch file names, expecting an integer formatter.
    :type file_template: str

    :returns: A list of aggregated data items from all batch files.
    :rtype: list
    """
    collected_data: List[Any] = []
    for i in range(num_batches):
        file_path = os.path.join(temp_dir, file_template.format(i))
        for item in load_from_pickle_generator(file_path):
            collected_data.append(item)
    return collected_data


def save_list_to_file(data_list: list, file_path: str) -> None:
    """
    Save a list to a file in JSON format.

    :param data_list: The list to save.
    :type data_list: list
    :param file_path: The path to the file where the list will be saved.
    :type file_path: str
    """
    with open(file_path, "w") as file:
        json.dump(data_list, file)


def load_list_from_file(file_path: str) -> list:
    """
    Load a list from a JSON-formatted file.

    :param file_path: The path to the file to read the list from.
    :type file_path: str

    :returns: The list loaded from the file.
    :rtype: list
    """
    with open(file_path, "r") as file:
        return json.load(file)


def save_dg(dg, path: str) -> str:
    """
    Save a DG instance to disk using MØD's dump method.

    :param dg: The derivation graph to save.
    :type dg: DG
    :param path: The file path where the graph will be dumped.
    :type path: str

    :returns: The path of the dumped file.
    :rtype: str

    :raises Exception: If saving fails.
    """
    try:
        dump_path = dg.dump(path)
        logger.info(f"DG saved to {dump_path}")
        return dump_path
    except Exception as e:
        logger.error(f"Error saving DG to {path}: {e}")
        raise


def load_dg(path: str, graph_db: list, rule_db: list):
    """
    Load a DG instance from a dumped file.

    :param path: The file path of the dumped graph.
    :type path: str
    :param graph_db: List of Graph objects representing the graph database.
    :type graph_db: list
    :param rule_db: List of Rule objects required for loading the DG.
    :type rule_db: list

    :returns: The loaded derivation graph instance.
    :rtype: DG

    :raises Exception: If loading fails.
    """
    from mod import DG

    try:
        dg = DG.load(graphDatabase=graph_db, ruleDatabase=rule_db, f=path)
        logger.info(f"DG loaded from {path}")
        return dg
    except Exception as e:
        logger.error(f"Error loading DG from {path}: {e}")
        raise
