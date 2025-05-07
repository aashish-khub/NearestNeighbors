#!/bin/bash

# Example usage:
# ./run_accuracy.sh OUTPUT_DIR

OUTPUT_DIR=$1
LOG_LEVEL=$2

METHODS=(
    "softimpute"
    "usvt"
    "row-row"
    "col-col"
    "dr"
    "ts"
)
for em in ${METHODS[@]};
do
    python run_scalar.py -od $OUTPUT_DIR -em $em --force --log_level $LOG_LEVEL --no-allow_self_neighbor
done
