#ifndef CLUSTERS_H
#define CLUSTERS_H
#include "ImageVectors.h"
//(Group continued from here)

//System: 
#include <vector>
#include <string>

using namespace std;

class Clusters {
private:
    int cluster_id;
    vector<ImageVectors> imageVectors;
    vector<double> cen;

public:
    Clusters(int clusterId, vector<double> centroid);
    void setCentroidByPos(int pos, double val);
    void setCentroid(vector<double> centroid);
    vector<double> getPoint(int pos) const;
    string returnLabel(int pos) const;
    void add(ImageVectors data);
    bool remove(int vectorId);
    int returnId() const;
    int returnSize() const;
    double getCentroidByPos(int pos) const;
    vector<double> returnCentroid() const;
};

#endif