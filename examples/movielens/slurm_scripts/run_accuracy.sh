# #!/bin/bash

# # Example usage:
# # ./run_accuracy.sh OUTPUT_DIR

# OUTPUT_DIR=$1

# METHODS=(
#     "usvt"
#     "row-row"
#     "col-col"
#     "dr"
#     "ts"
# )
# for em in ${METHODS[@]};
# do
#     python run_scalar.py -od $OUTPUT_DIR -em $em --force
# done
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
    "softimpute"
)
for em in ${METHODS[@]};
do
    python run_scalar.py -od $OUTPUT_DIR -em $em --force
done