#!/bin/bash

# Example usage:
# ./run_accuracy.sh OUTPUT_DIR

OUTPUT_DIR=$1

METHODS=(
    "usvt"
    "row-row"
    "col-col"
    "dr"
    "ts"
    "aw"
)
for em in ${METHODS[@]};
do
    python run_scalar.py -od $OUTPUT_DIR -em $em
done
