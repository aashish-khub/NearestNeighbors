"""Script to evaluate NN methods on HeartSteps data."""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--method', '-m', type=str, default='vanilla')
# TODO: add remaining arguments
args = parser.parse_args()