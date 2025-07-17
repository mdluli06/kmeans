#ifndef KMEANS_H
#define KMEANS_H
#include "Clusters.h"
#include "ImageVectors.h"
#include "opencv2/opencv.hpp"
//#ifdef USE_MPI
#include <mpi.h>
//#endif
#include <vector>
#include <map>
#include <string>

using namespace std;
using namespace cv;

class Kmeans {
private:
    vector<Clusters> clusters;
    int n_clusters, n_points;
    int rank, size; // MPI rank and size
    void computeCentroids();
    double computeDistance(vector<double> point_a, vector<double> point_b) const;
    vector<double> Sum(vector<double> point_a, vector<double> point_b) const;
    int getNearestClusterId(ImageVectors vector) const;
    vector<map<string, int>> mapCluster() const;
    float accuracy(vector<int> predictions, vector<int> labels) const;
    void sendCentroidsToServer(); // Send local centroids to server
    void receiveCentroidsFromServer(); //  Receive global centroids
    void federatedAverage(vector<double>& global_centroids, int num_workers); // Server aggregates centroids

public:
    Kmeans(int K, int rank, int size);
    void run(vector<ImageVectors>& all_vectors, int max_iterations = 100);
    void output();
};

#endif