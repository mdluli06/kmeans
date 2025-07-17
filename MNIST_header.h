#ifndef MNIST_HEADER_H
#define MNIST_HEADER_H
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

using namespace std;
using namespace cv;

struct Image {
    Mat image;
    string label;
};

class MNIST {
public:
    MNIST(string train_image, string train_label, string test_image, string test_label, int rank);
    void run();
    vector<Image>& returnTest();
    vector<Image>& returnTrain();
    void applyTransformation(int rotation_angle); // Apply rotation to images
private:
    vector<Image> train_set;
    vector<Image> test_set;
    string trainImg, trainLabel, testImg, testLabel;
    int n_rows, n_cols;
    int rank; // MPI rank for worker-specific transformations
    uint32_t reverse_int(uint32_t val);
    void mnist_reader(const char* image_filename, const char* label_filename, vector<Image>& dataset, int start_idx, int num_images);
};

#endif