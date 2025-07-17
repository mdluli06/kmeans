#include "ImageVectors.h"

ImageVectors::ImageVectors(Mat flattened, string labels) {
    Mat temp;
    flattened.convertTo(temp, CV_64F, 1.0 / 255.0); // Normalize 
    values.assign((double*)temp.datastart, (double*)temp.dataend);
    Label = labels;
    clusterId = -99;
    vecId = -99;
}

void ImageVectors::setClusterId(int val) {
    clusterId = val;
}

void ImageVectors::setId(int val) {
    vecId = val;
}

int ImageVectors::returnClusterId() const {
    return clusterId;
}

int ImageVectors::returnId() const {
    return vecId;
}

double ImageVectors::returnVal(int pos) const {
    if (pos < 0 || pos >= values.size()) throw std::out_of_range("Index out of bounds");
    return values[pos];
}

const vector<double>& ImageVectors::returnPoints() const {
    return values;
}

string ImageVectors::returnLabel() const {
    return Label;
}