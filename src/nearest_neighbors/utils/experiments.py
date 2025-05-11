"""Utility functions for the experiment scripts"""

from argparse import ArgumentParser
import logging


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
        choices=["dr", "ts", "row-row", "col-col", "usvt", "star", "softimpute"],
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
