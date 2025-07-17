#!/bin/bash

# run.sh: Script to execute the federated k-means clustering program with MPI on multiple nodes

# Number of MPI processes (1 server + 3 workers)
NUM_PROCESSES=4

# Path to executable
EXEC=./kmeans_federated

# Path to MNIST dataset
DATA_DIR=./mnistdataset

# Path to hostfile listing nodes and slots
HOSTFILE=./hosts.txt

# Check if executable exists
if [ ! -f "$EXEC" ]; then
    echo "Error: Executable $EXEC not found. Run 'make' to build."
    exit 1
fi

# Check if MNIST dataset directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: MNIST dataset directory $DATA_DIR not found."
    echo "Download MNIST dataset and place in $DATA_DIR."
    exit 1
fi

# Check for required MNIST files
required_files=(
    "$DATA_DIR/train-images.idx3-ubyte"
    "$DATA_DIR/train-labels.idx1-ubyte"
    "$DATA_DIR/t10k-images.idx3-ubyte"
    "$DATA_DIR/t10k-labels.idx1-ubyte"
)
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found in $DATA_DIR."
        exit 1
    fi
done

# Check if hostfile exists
if [ ! -f "$HOSTFILE" ]; then
    echo "Error: Hostfile $HOSTFILE not found."
    echo "Create a hostfile with node names and slots, e.g.:"
    echo "node1 slots=2"
    echo "node2 slots=2"
    exit 1
fi

# Run the program with MPI using the hostfile to span multiple nodes
echo "Running federated k-means with $NUM_PROCESSES processes across multiple nodes..."
mpirun -np $NUM_PROCESSES --hostfile $HOSTFILE $EXEC

if [ $? -eq 0 ]; then
    echo "Execution completed successfully."
else
    echo "Error: Execution failed."
    exit 1
fi
