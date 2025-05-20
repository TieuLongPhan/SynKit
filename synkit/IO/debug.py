import os
import logging
import warnings
from rdkit import rdBase, RDLogger


def setup_logging(
    log_level: str = "INFO", log_filename: str = None, task_type: str = None
) -> logging.Logger:
    """
    Configures logging to either the console or a file, based on provided parameters.

    :param log_level: Logging level to set. Defaults to 'INFO'.
                      Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    :type log_level: str
    :param log_filename: If provided, logs are written to this file. Defaults to None (logs to console).
    :type log_filename: str or None
    :param task_type: Logger name/namespace. Useful for distinguishing loggers in multi-task settings.
                     Defaults to None.
    :type task_type: str or None

    :returns: Configured logger instance.
    :rtype: logging.Logger

    :raises ValueError: If an invalid log level is provided.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger(task_type)
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

    :param ignore_warnings: Whether to suppress all Python warnings. Default is False.
    :type ignore_warnings: bool
    :param disable_rdkit_logs: Whether to disable RDKit error and warning logs. Default is False.
    :type disable_rdkit_logs: bool

    :returns: None

    :usage:
        Use this function to control verbosity (e.g. in production or testing), but use with
        caution during development to avoid missing critical issues.
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
