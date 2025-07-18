#!/bin/bash

# Example usage:

OUTPUT_DIR="out"
LOG_LEVEL="WARNING"
EXPERIMENT="all"


usage() {
    echo "Usage: $0 [-o OUTPUT_DIR] [-l LOG_LEVEL] [-e EXPERIMENT]"
    exit 1
}

while getopts ":o:l:e:" opt; do
    case ${opt} in
        o ) OUTPUT_DIR=$OPTARG ;;
        l ) LOG_LEVEL=$OPTARG ;;
        e ) EXPERIMENT=$OPTARG ;;
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
echo $EXPERIMENT

if [ "$EXPERIMENT" = "all" ]
then
    EXPERIMENTS=("heartsteps" "movielens" "prompteval" "prop99")
else
    EXPERIMENTS=($EXPERIMENT)
fi

for exper in ${EXPERIMENTS[@]};
do
    if [ "$exper" = "heartsteps" ] 
    then
        HEARTSTEPS_DIR="../../bench/${OUTPUT_DIR}/heartsteps"
        echo "Running heartsteps experiment"
        cd ../examples/heartsteps
        ./slurm_scripts/run_accuracy.sh $HEARTSTEPS_DIR $LOG_LEVEL
        python run_distribution.py -od $HEARTSTEPS_DIR -dt kernel_mmd --force --log_level $LOG_LEVEL -em col-col
        python run_distribution.py -od $HEARTSTEPS_DIR -dt wasserstein_samples --force --log_level $LOG_LEVEL -em col-col
        cd ../../bench
    elif [ "$exper" = "movielens" ] 
    then
        MOVIELENS_DIR="../../bench/${OUTPUT_DIR}/movielens"
        echo "Running movielens experiment"
        cd ../examples/movielens
        ./slurm_scripts/run_accuracy.sh $MOVIELENS_DIR $LOG_LEVEL
        cd ../../bench
    elif [ "$exper" = "prompteval" ] 
    then
        PROMPTEVAL_DIR="../../bench/${OUTPUT_DIR}/prompteval"
        echo "Running prompteval experiment"
        cd ../examples/prompteval
        ./slurm_scripts/run_accuracy.sh $PROMPTEVAL_DIR $LOG_LEVEL
        python run_distribution.py -od $PROMPTEVAL_DIR -dt kernel_mmd --force --log_level $LOG_LEVEL -em col-col
        python run_distribution.py -od $PROMPTEVAL_DIR -dt wasserstein_samples --force --log_level $LOG_LEVEL -em col-col
        cd ../../bench
    elif [ "$exper" = "prop99" ] 
    then
        PROP99_DIR="../../bench/${OUTPUT_DIR}/prop99"
        cd ../examples/prop99
        echo "Running prop99 experiment"
        ./slurm_scripts/run_accuracy.sh $PROP99_DIR $LOG_LEVEL
        ./slurm_scripts/run_california.sh $PROP99_DIR $LOG_LEVEL
        python proposal_99.py -od $OUTPUT_DIR --force --log_level $LOG_LEVEL
        cd ../../bench
    else
        echo "$exper is not a valid experiment. Please choose from heartsteps, movielens, prompteval, or prop99."
    fi
done
# SIM_METHODS=(
#     "auto"
#     "dr"
#     "ts"
# )
# SIM_OUTPUT_DIR="${OUTPUT_DIR}/simulations"
# for em in ${SIM_METHODS[@]};
# do
#     python ../examples/simulations/run_scalar.py -od "${SIM_OUTPUT_DIR}/high_snr" -em $em --force --log_level $LOG_LEVEL -nstd 0.001
# done

# for em in ${SIMMETHODS[@]};
# do
#     python ../examples/simulations/run_scalar.py -od ${SIM_OUTPUT_DIR}/low_snr -em $em --force --log_level $LOG_LEVEL -nstd 1.0
# done
