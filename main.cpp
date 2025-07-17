#include "MNIST_header.h"
#include "Kmeans.h"
#include "BaseKmeans.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) cerr << "At least 2 processes required (1 server, 1+ workers)" << endl;
        MPI_Finalize();
        return 1;
    }

    string HOME_DIR = "../mnistdataset/";
    string train_image = HOME_DIR + "train-images.idx3-ubyte";
    string train_label = HOME_DIR + "train-labels.idx1-ubyte";
    string test_image = HOME_DIR + "t10k-images.idx3-ubyte";
    string test_label = HOME_DIR + "t10k-labels.idx1-ubyte";

    // Create MNIST instance with rank and size
    MNIST mnist(train_image, train_label, test_image, test_label, rank, size);

    if (rank != 0) {
        mnist.run();
        int rotation_angle = ((rank - 1) % 4) * 90;
        mnist.applyTransformation(rotation_angle);
    }

    // Convert training images to ImageVectors
    vector<ImageVectors> vectors;
    if (rank != 0) {
        vector<Image> train = mnist.returnTrain();
        for (size_t i = 0; i < train.size(); ++i) {
            Mat flat = train[i].image.reshape(1, 1);
            flat.convertTo(flat, CV_64F, 1.0 / 255.0);
            vectors.emplace_back(flat, train[i].label);
            vectors.back().setId(i);
        }
    }

    // Run federated K-means
    if (rank == 0) {
        cout << "Running federated K-means..." << endl;
    }
    Kmeans Kmeans(10, rank, size);
    Kmeans.run(vectors);
    Kmeans.output();

    // Run baseline centralized K-means on rank 0
    if (rank == 0) {
        cout << "\nRunning baseline centralized K-means..." << endl;
        
        // Load full dataset for baseline
        MNIST full_mnist(train_image, train_label, test_image, test_label, rank);
        full_mnist.run();
        
        vector<ImageVectors> full_vectors;
        vector<Image> full_train = full_mnist.returnTrain();
        for (size_t i = 0; i < full_train.size(); ++i) {
            Mat flat = full_train[i].image.reshape(1, 1);
            flat.convertTo(flat, CV_64F, 1.0 / 255.0);
            full_vectors.emplace_back(flat, full_train[i].label);
            full_vectors.back().setId(i);
        }
        
        BaseKmeans baseKmeans(10);
        baseKmeans.run(full_vectors);
        baseKmeans.output();
    }

    MPI_Finalize();
    return 0;
}