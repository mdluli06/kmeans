#include "BaseKmeans.h"
#include <iostream>

BaseKmeans::BaseKmeans(int K) {
    n_clusters = K;
}

double BaseKmeans::computeDistance(std::vector<double> point_a, std::vector<double> point_b) const {
    if (point_a.size() != point_b.size()) throw std::runtime_error("Vector size mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < point_a.size(); i++) {
        sum += (point_a[i] - point_b[i]) * (point_a[i] - point_b[i]);
    }
    return sqrt(sum);
}

std::vector<double> BaseKmeans::Sum(std::vector<double> point_a, std::vector<double> point_b) const {
    if (point_a.size() != point_b.size()) throw std::runtime_error("Vector size mismatch");
    std::vector<double> result = point_a;
    for (size_t i = 0; i < result.size(); i++) {
        result[i] += point_b[i];
    }
    return result;
}

void BaseKmeans::computeCentroids() {
    for (size_t i = 0; i < clusters.size(); i++) {
        if (clusters[i].returnSize() == 0) continue; // Skip empty clusters
        std::vector<double> tempCentroid(clusters[i].returnCentroid().size(), 0.0);
        for (int j = 0; j < clusters[i].returnSize(); j++) {
            tempCentroid = Sum(tempCentroid, clusters[i].getPoint(j));
        }
        for (size_t j = 0; j < tempCentroid.size(); j++) {
            tempCentroid[j] /= clusters[i].returnSize();
        }
        clusters[i].setCentroid(tempCentroid);
    }
}

int BaseKmeans::getNearestClusterId(ImageVectors vector) const {
    double minDistance = std::numeric_limits<double>::max();
    int clusterId = -1;
    for (size_t i = 0; i < clusters.size(); i++) {
        double distance = computeDistance(vector.returnPoints(), clusters[i].returnCentroid());
        if (distance < minDistance) {
            minDistance = distance;
            clusterId = clusters[i].returnId();
        }
    }
    return clusterId;
}

void BaseKmeans::run(std::vector<ImageVectors>& all_vectors, int max_iterations) {
    n_points = all_vectors.size();
    
    // Initialize clusters with random centroids (K-means++)
    std::vector<int> usedVectorIds;
    for (int i = 0; i < n_clusters; i++) {
        while (true) {
            int index = rand() % n_points;
            if (std::find(usedVectorIds.begin(), usedVectorIds.end(), index) == usedVectorIds.end()) {
                usedVectorIds.push_back(index);
                Clusters tempCluster(i, all_vectors[index].returnPoints());
                clusters.push_back(tempCluster);
                break;
            }
        }
    }

    // K-means iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        bool changed = false;
        
        // Assign points to nearest clusters
        for (int j = 0; j < n_points; j++) {
            int currentClusterId = all_vectors[j].returnClusterId();
            int nearestClusterId = getNearestClusterId(all_vectors[j]);
            
            if (currentClusterId != nearestClusterId) {
                changed = true;
                
                // Remove from old cluster
                for (size_t k = 0; k < clusters.size(); k++) {
                    if (clusters[k].returnId() == currentClusterId) {
                        clusters[k].remove(all_vectors[j].returnId());
                    }
                }
                
                // Add to new cluster
                for (size_t k = 0; k < clusters.size(); k++) {
                    if (clusters[k].returnId() == nearestClusterId) {
                        clusters[k].add(all_vectors[j]);
                    }
                }
                
                all_vectors[j].setClusterId(nearestClusterId);
            }
        }
        
        // Recompute centroids
        computeCentroids();
        
        // Early stopping if no changes
        if (!changed) break;
    }
}

std::vector<std::map<std::string, int>> BaseKmeans::mapCluster() const {
    std::vector<std::map<std::string, int>> countTable;
    for (int i = 0; i < n_clusters; i++) {
        std::map<std::string, int> LabelCount;
        for (int j = 0; j < clusters[i].returnSize(); j++) {
            std::string label = clusters[i].returnLabel(j);
            LabelCount[label]++;
        }
        countTable.push_back(LabelCount);
    }
    return countTable;
}

float BaseKmeans::accuracy(std::vector<int> predictions, std::vector<int> labels) const {
    if (predictions.size() != labels.size()) throw std::runtime_error("Prediction/label size mismatch");
    int count = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == labels[i]) count++;
    }
    return (100.0 * count) / predictions.size();
}

void BaseKmeans::output() {
    std::vector<std::map<std::string, int>> countTable = mapCluster();
    std::map<int, int> clusterIdToImageLabelMap;
    
    for (size_t i = 0; i < countTable.size(); i++) {
        std::cout << "Cluster:" << clusters[i].returnId() << std::endl;
        std::cout << "Numbers count = " << std::endl;
        int maxImageLabel = -99;
        int maxCount = 0;
        for (auto itr = countTable[i].begin(); itr != countTable[i].end(); ++itr) {
            std::cout << itr->first << "=" << itr->second << ", ";
            if (itr->second > maxCount) {
                maxCount = itr->second;
                maxImageLabel = std::stoi(itr->first);
            }
        }
        clusterIdToImageLabelMap[clusters[i].returnId()] = maxImageLabel;
        std::cout << std::endl;
    }
    
    std::vector<int> predictions, imageLabels;
    for (size_t i = 0; i < clusters.size(); i++) {
        for (int j = 0; j < clusters[i].returnSize(); j++) {
            int label = std::stoi(clusters[i].returnLabel(j));
            int prediction = clusterIdToImageLabelMap[clusters[i].returnId()];
            predictions.push_back(prediction);
            imageLabels.push_back(label);
        }
    }
    
    std::cout << "Accuracy = " << accuracy(predictions, imageLabels) << "%" << std::endl;
}