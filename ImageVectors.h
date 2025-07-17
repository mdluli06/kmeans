#ifndef IMAGEVECTORS_H
#define IMAGEVECTORS_H
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

using namespace std;
using namespace cv;

class ImageVectors {
private:
    int vecId, clusterId;
    string Label;
    vector<double> values;

public:
    ImageVectors(Mat flattened, string labels);
    const vector<double>& returnPoints() const;
    string returnLabel() const;
    int returnId() const;
    int returnClusterId() const;
    void setClusterId(int val);
    void setId(int val);
    double returnVal(int pos) const;
};

#endif