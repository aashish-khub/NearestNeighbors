"""Utility functions for the experiment scripts"""

from argparse import ArgumentParser
import logging
import numpy as np


def get_base_parser() -> ArgumentParser:
    """Get the base CLI parser for the heartsteps scripts"""
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", "-od", type=str, default="out", help="Output directory"
    )
    parser.add_argument(
        "--estimation_method",
        "-em",
        type=str,
        default="row-row",
        choices=[
            "dr",
            "ts",
            "row-row",
            "col-col",
            "usvt",
            "softimpute",
            "auto",
            "star",
            "usvt",
            "tabpfn",
        ],
        help="Estimation method to use",
    )
    parser.add_argument(
        "--fit_method",
        "-fm",
        type=str,
        default="lbo",
        choices=["dr", "ts", "lbo", "usvt"],
        help="Fit method to use",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite of existing results",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "--allow_self_neighbor", action="store_true", help="Allow self neighbor"
    )
    parser.add_argument(
        "--raw_threshold",
        action="store_true",
        help="Use raw (not percentile-based) for distance threshold",
    )

    parser.add_argument(
        "--propensity",
        "-p",
        type=float,
        default=0.5,
        help="Propensity for the missing data",
    )
    return parser


def setup_logging(log_level: str) -> None:
    """Setup logging for the experiment scripts

    Args:
        log_level (str): Log level to use

    """
    # Suppress httpx logging from API requests
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # need to silence entrywise
    logging.getLogger("hyperopt").setLevel(logging.WARNING)  # or logging.ERROR


def serialize(x: np.ndarray) -> list:
    """Serialize a numpy array to a list.

    Args:
        x (np.ndarray): The numpy array to serialize.

    Returns:
        list: The serialized numpy array.

    """
    if np.any(np.isnan(x)):
        return []
    else:
        return x.tolist()  # type: ignore
