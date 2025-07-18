#!/bin/bash

# Example usage:
# ./run_accuracy.sh OUTPUT_DIR LOG_LEVEL NOISE_LEVEL

OUTPUT_DIR="out"
LOG_LEVEL="WARNING"
NOISE_SIGMA=0.001

usage() {
    echo "Usage: $0 [-o OUTPUT_DIR] [-l LOG_LEVEL] [-n NOISE_SIGMA]"
    exit 1
}

while getopts ":o:l:n:" opt; do
    case ${opt} in
        o ) OUTPUT_DIR=$OPTARG ;;
        l ) LOG_LEVEL=$OPTARG ;;
        n ) NOISE_SIGMA=$OPTARG ;;
        \? )
        echo "Invalid option: -$OPTARG" 1>&2
        usage
        ;;
        : )
        echo "Invalid option: -$OPTARG requires an argument" 1>&2
        usage
        ;;
    esac
done
shift $((OPTIND -1))

METHODS=(
    # "softimpute"
    # "usvt"
    "auto"
    #"row-row"
    #"col-col"
    "dr"
    "ts"
    #"softimpute"

)


for em in ${METHODS[@]};
do
    python run_scalar.py -od $OUTPUT_DIR -em $em --force --log_level $LOG_LEVEL -nstd $NOISE_SIGMA
done
