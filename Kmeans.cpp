#include "Kmeans.h"
#include <algorithm>
#include <limits>
#include <numeric>

Kmeans::Kmeans(int K, int rank, int size) {
    n_clusters = K;
    this->rank = rank;
    this->size = size;
}

double Kmeans::computeDistance(vector<double> point_a, vector<double> point_b) const {
    if (point_a.size() != point_b.size()) throw std::runtime_error("Vector size mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < point_a.size(); i++) {
        sum += (point_a[i] - point_b[i]) * (point_a[i] - point_b[i]);
    }
    return sqrt(sum);
}

vector<double> Kmeans::Sum(vector<double> point_a, vector<double> point_b) const {
    if (point_a.size() != point_b.size()) throw std::runtime_error("Vector size mismatch");
    vector<double> result = point_a;
    for (size_t i = 0; i < result.size(); i++) {
        result[i] += point_b[i];
    }
    return result;
}

void Kmeans::computeCentroids() {
    for (size_t i = 0; i < clusters.size(); i++) {
        if (clusters[i].returnSize() == 0) continue; // Skip empty clusters
        vector<double> tempCentroid(clusters[i].returnCentroid().size(), 0.0);
        for (int j = 0; j < clusters[i].returnSize(); j++) {
            tempCentroid = Sum(tempCentroid, clusters[i].getPoint(j));
        }
        for (size_t j = 0; j < tempCentroid.size(); j++) {
            tempCentroid[j] /= clusters[i].returnSize();
        }
        clusters[i].setCentroid(tempCentroid);
    }
}

int Kmeans::getNearestClusterId(ImageVectors vector) const {
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

void Kmeans::sendCentroidsToServer() {
    // Pack all centroids into a single buffer
    int centroid_size = clusters[0].returnCentroid().size();
    vector<double> buffer(n_clusters * centroid_size);
    for (int i = 0; i < n_clusters; i++) {
        vector<double> centroid = clusters[i].returnCentroid();
        for (int j = 0; j < centroid_size; j++) {
            buffer[i * centroid_size + j] = centroid[j];
        }
    }
    MPI_Send(buffer.data(), n_clusters * centroid_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

void Kmeans::receiveCentroidsFromServer() {
    int centroid_size = clusters[0].returnCentroid().size();
    vector<double> buffer(n_clusters * centroid_size);
    MPI_Bcast(buffer.data(), n_clusters * centroid_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < n_clusters; i++) {
        vector<double> centroid(centroid_size);
        for (int j = 0; j < centroid_size; j++) {
            centroid[j] = buffer[i * centroid_size + j];
        }
        clusters[i].setCentroid(centroid);
    }
}

void Kmeans::federatedAverage(vector<double>& global_centroids, int num_workers) {
    int centroid_size = clusters[0].returnCentroid().size();
    vector<double> sum_centroids(n_clusters * centroid_size, 0.0);
    vector<double> local_buffer(n_clusters * centroid_size);
    
    // Server receives centroids from all workers
    for (int i = 1; i < num_workers; i++) {
        MPI_Recv(local_buffer.data(), n_clusters * centroid_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (size_t j = 0; j < local_buffer.size(); j++) {
            sum_centroids[j] += local_buffer[j];
        }
    }
    // Add server's own centroids (if server also trains)
    for (int i = 0; i < n_clusters; i++) {
        vector<double> centroid = clusters[i].returnCentroid();
        for (int j = 0; j < centroid_size; j++) {
            sum_centroids[i * centroid_size + j] += centroid[j];
        }
    }
    // Average centroids
    for (size_t i = 0; i < sum_centroids.size(); i++) {
        global_centroids[i] = sum_centroids[i] / num_workers;
    }
}

void Kmeans::run(vector<ImageVectors>& all_vectors, int max_iterations) {
    n_points = all_vectors.size();
    // Initialize clusters with random centroids (K-means++)
    vector<int> usedVectorIds;
    for (int i = 0; i < n_clusters; i++) {
        while (true) {
            int index = rand() % n_points;
            if (find(usedVectorIds.begin(), usedVectorIds.end(), index) == usedVectorIds.end()) {
                usedVectorIds.push_back(index);
                Clusters tempCluster(i, all_vectors[index].returnPoints());
                clusters.push_back(tempCluster);
                break;
            }
        }
    }

    // Local K-means iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        bool changed = false;
        for (int j = 0; j < n_points; j++) {
            int currentClusterId = all_vectors[j].returnClusterId();
            int nearestClusterId = getNearestClusterId(all_vectors[j]);
            if (currentClusterId != nearestClusterId) {
                changed = true;
                for (size_t k = 0; k < clusters.size(); k++) {
                    if (clusters[k].returnId() == currentClusterId) {
                        clusters[k].remove(all_vectors[j].returnId());
                    }
                }
                for (size_t k = 0; k < clusters.size(); k++) {
                    if (clusters[k].returnId() == nearestClusterId) {
                        clusters[k].add(all_vectors[j]);
                    }
                }
                all_vectors[j].setClusterId(nearestClusterId);
            }
        }
        computeCentroids();
        if (!changed) break;

        // Federated averaging
        if (rank == 0) {
            vector<double> global_centroids(n_clusters * clusters[0].returnCentroid().size());
            federatedAverage(global_centroids, size);
            MPI_Bcast(global_centroids.data(), global_centroids.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (int i = 0; i < n_clusters; i++) {
                vector<double> centroid(clusters[i].returnCentroid().size());
                for (size_t j = 0; j < centroid.size(); j++) {
                    centroid[j] = global_centroids[i * centroid.size() + j];
                }
                clusters[i].setCentroid(centroid);
            }
        } else {
            sendCentroidsToServer();
            receiveCentroidsFromServer();
        }
    }
}

void Kmeans::output() {
    if (rank != 0) return; // Only server outputs
    vector<map<string, int>> countTable = mapCluster();
    map<int, int> clusterIdToImageLabelMap;
    for (size_t i = 0; i < countTable.size(); i++) {
        cout << "Cluster:" << clusters[i].returnId() << endl;
        cout << "Numbers count = " << endl;
        int maxImageLabel = -99;
        int maxCount = 0;
        for (auto itr = countTable[i].begin(); itr != countTable[i].end(); ++itr) {
            cout << itr->first << "=" << itr->second << ", ";
            if (itr->second > maxCount) {
                maxCount = itr->second;
                maxImageLabel = stoi(itr->first);
            }
        }
        clusterIdToImageLabelMap[clusters[i].returnId()] = maxImageLabel;
        cout << endl;
    }
    vector<int> predictions, imageLabels;
    for (size_t i = 0; i < clusters.size(); i++) {
        for (int j = 0; j < clusters[i].returnSize(); j++) {
            int label = stoi(clusters[i].returnLabel(j));
            int prediction = clusterIdToImageLabelMap[clusters[i].returnId()];
            predictions.push_back(prediction);
            imageLabels.push_back(label);
        }
    }
    cout << "Accuracy = " << accuracy(predictions, imageLabels) << "%" << endl;
}

vector<map<string, int>> Kmeans::mapCluster() const {
    vector<map<string, int>> countTable;
    for (int i = 0; i < n_clusters; i++) {
        map<string, int> LabelCount;
        for (int j = 0; j < clusters[i].returnSize(); j++) {
            string label = clusters[i].returnLabel(j);
            LabelCount[label]++;
        }
        countTable.push_back(LabelCount);
    }
    return countTable;
}

float Kmeans::accuracy(vector<int> predictions, vector<int> labels) const {
    if (predictions.size() != labels.size()) throw std::runtime_error("Prediction/label size mismatch");
    int count = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == labels[i]) count++;
    }
    return (100.0 * count) / predictions.size();
}