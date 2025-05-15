#!/bin/bash

# Example usage:
# ./slurm_scripts/run_distribution.sh OUTPUT_DIR

OUTPUT_DIR=$1

METHODS=(
    "row-row"
    "col-col"
)

DATA_TYPE=(
    "wasserstein_samples"
    "kernel_mmd"
)

ps=(
    "0.1"
    "0.5"
    "0.9"
)

for data_type in ${DATA_TYPE[@]};
do
    for em in ${METHODS[@]};
    do
        for p in ${ps[@]};
        do
            echo $data_type
            CMD="python run_distribution.py -em $em -p $p -tp 4.0 --data_type $data_type -s 1 --force"
            echo $CMD
            eval $CMD
        done
    done
done
