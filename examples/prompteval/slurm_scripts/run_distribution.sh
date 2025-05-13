#!/bin/bash

# Example usage:
# ./slurm_scripts/run_distribution.sh OUTPUT_DIR

OUTPUT_DIR=$1

METHODS=(
    "row-row"
    "col-col"
)
for em in ${METHODS[@]};
do
    for p in $(seq 0.1 0.1 1.0);
    do
        CMD="python run_distribution.py -od $OUTPUT_DIR -em $em -p $p -tp 4 -s 1 -f"
        echo $CMD
        eval $CMD
    done
done
