#!/bin/bash

# Example usage:
# ./run_accuracy.sh OUTPUT_DIR

OUTPUT_DIR=$1
LOG_LEVEL=$2

METHODS=(
    # "softimpute"
    # "usvt"
    "auto"
    "row-row"
    "col-col"
    "dr"
    "ts"
    "softimpute"
    
)
for em in ${METHODS[@]};
do
    python run_scalar.py -od $OUTPUT_DIR -em $em --force --log_level $LOG_LEVEL
done
