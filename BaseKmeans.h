#ifndef BASEKMEANS_H
#define BASEKMEANS_H

#include "MNIST_header.h"
#include <vector>
#include <map>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cmath>

class BaseKmeans {
private:
    int n_clusters;
    int n_points;
    std::vector<Clusters> clusters;

    double computeDistance(std::vector<double> point_a, std::vector<double> point_b) const;
    std::vector<double> Sum(std::vector<double> point_a, std::vector<double> point_b) const;
    void computeCentroids();
    int getNearestClusterId(ImageVectors vector) const;
    std::vector<std::map<std::string, int>> mapCluster() const;
    float accuracy(std::vector<int> predictions, std::vector<int> labels) const;

public:
    BaseKmeans(int K);
    void run(std::vector<ImageVectors>& all_vectors, int max_iterations = 100);
    void output();
};

#endif 