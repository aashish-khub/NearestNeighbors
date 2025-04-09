"""Utility functions for the heartsteps scripts"""

from argparse import ArgumentParser


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
        default="vanilla",
        choices=["dr", "ts", "vanilla", "usvt"],
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
    return parser
