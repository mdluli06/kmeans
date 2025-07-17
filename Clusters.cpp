#include "Clusters.h"

Clusters::Clusters(int clusterId, vector<double> centroid) {
    cluster_id = clusterId;
    cen = centroid;
}

void Clusters::setCentroidByPos(int pos, double val) {
    if (pos < 0 || pos >= cen.size()) throw std::out_of_range("Centroid index out of bounds");
    cen[pos] = val;
}

vector<double> Clusters::getPoint(int pos) const {
    if (pos < 0 || pos >= imageVectors.size()) throw std::out_of_range("Vector index out of bounds");
    return imageVectors[pos].returnPoints();
}

string Clusters::returnLabel(int pos) const {
    if (pos < 0 || pos >= imageVectors.size()) throw std::out_of_range("Vector index out of bounds");
    return imageVectors[pos].returnLabel();
}

void Clusters::add(ImageVectors data) {
    data.setClusterId(cluster_id);
    imageVectors.push_back(data);
}

bool Clusters::remove(int vectorId) {
    for (size_t i = 0; i < imageVectors.size(); i++) {
        if (imageVectors[i].returnId() == vectorId) {
            imageVectors.erase(imageVectors.begin() + i);
            return true;
        }
    }
    return false;
}

int Clusters::returnId() const {
    return cluster_id;
}

int Clusters::returnSize() const {
    return imageVectors.size();
}

double Clusters::getCentroidByPos(int pos) const {
    if (pos < 0 || pos >= cen.size()) throw std::out_of_range("Centroid index out of bounds");
    return cen[pos];
}

void Clusters::setCentroid(vector<double> centroid) {
    cen = centroid;
}

vector<double> Clusters::returnCentroid() const {
    return cen;
}