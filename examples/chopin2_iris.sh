#!/bin/bash
# Test chopin2.py over the iris dataset of the scikit-learn package

# Define the script root folder
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Define the iris.csv URL pointing to the sklearn repository on GitHub
DATASET_URL="https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/iris.csv"

# Define the path to the iris.csv dataset on the file system
DATASET_PATH=$ROOT/iris.csv

# Download the iris.csv dataset with curl
curl -o ${DATASET_PATH} ${DATASET_URL}

# Run chopin2.py on the iris.csv dataset
# The python script is located under the same folder of this bash script
python $ROOT/chopin2.py --input ${DATASET_PATH} \
                        --fieldsep , \
                        --dimensionality 10000 \
                        --levels 10 \
                        --feature-selection backward \
                        --kfolds 5 \
                        --retrain 10 \
                        --nproc 2