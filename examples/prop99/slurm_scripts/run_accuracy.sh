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
    "star"
    "softimpute"
)
CONTROL_STATES=(
    "TX" "WI" "MT" "RI" "KS" "ME" "UT" "VA" "IN" "GA"
    "AL" "SD" "NH" "DE" "NC" "CO" "AR" "CT" "MN" "NV"
    "LA" "IA" "NE" "SC" "OH" "TN" "WV" "KY" "ID" "MS"
    "IL" "WY" "VT" "ND" "PA" "OK" "MO" "NM"
)
for state in ${CONTROL_STATES[@]};
do
    for em in ${METHODS[@]};
    do
        python run_scalar.py -od $OUTPUT_DIR -em $em -f --state $state --log_level ERROR
    done
done
