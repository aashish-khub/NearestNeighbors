#!/bin/bash

# Example usage:
# ./run_accuracy.sh OUTPUT_DIR

OUTPUT_DIR=$1
for em in usvt row-row col-col dr ts;
do
    python run_scalar.py -od $OUTPUT_DIR -em $em
done
