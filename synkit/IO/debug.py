import os
import logging
import warnings
from rdkit import rdBase, RDLogger


def setup_logging(log_level: str = "INFO", log_filename: str = None) -> logging.Logger:
    """
    Configures logging to either the console or a file based on provided parameters.

    Parameters
    ----------
    log_level : str, optional
        Logging level to set. Defaults to 'INFO'. Options include 'DEBUG', 'INFO',
        'WARNING', 'ERROR', 'CRITICAL'.
    log_filename : str, optional
        If provided, logs are written to this file. Defaults to None,
        which logs to console.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Raises
    ------
    ValueError
        If an invalid log level is provided.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger()
    logger.handlers.clear()  # Efficiently remove all existing handlers

    if log_filename:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(
            level=numeric_level, format=log_format, filename=log_filename, filemode="a"
        )
    else:
        logging.basicConfig(level=numeric_level, format=log_format)

    return logger


def configure_warnings_and_logs(
    ignore_warnings: bool = False, disable_rdkit_logs: bool = False
) -> None:
    """
    Configures Python warnings and RDKit log behavior based on input flags.

    Parameters
    ----------
    ignore_warnings : bool, optional
        Whether to suppress Python warnings. Default is False.
    disable_rdkit_logs : bool, optional
        Whether to disable RDKit error and warning logs. Default is False.

    Usage
    -----
    This function is useful for controlling verbosity in production or testing, but
    should be used cautiously during development to avoid missing critical issues.
    """
    if ignore_warnings:
        warnings.filterwarnings("ignore")
    else:
        warnings.resetwarnings()

    if disable_rdkit_logs:
        rdBase.DisableLog("rdApp.error")
        rdBase.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.warning")
    else:
        rdBase.EnableLog("rdApp.error")
        rdBase.EnableLog("rdApp.warning")
