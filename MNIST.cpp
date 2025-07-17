#include "MNIST_header.h"
#include <fstream>
#include <iostream>

MNIST::MNIST(string train_image, string train_label, string test_image, string test_label, int rank, int size) {
    trainImg = train_image;
    trainLabel = train_label;
    testImg = test_image;
    testLabel = test_label;
    this->rank = rank;
    this->size = size;    // set size member
}

uint32_t MNIST::reverse_int(uint32_t i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

void MNIST::mnist_reader(const char* image_filename, const char* label_filename, vector<Image>& container, int start_idx, int num_images) {
    std::ifstream image_file(image_filename, std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::binary);
    if (!image_file.is_open() || !label_file.is_open()) {
        throw std::runtime_error("Failed to open MNIST files");
    }

    uint32_t magic, num_items, num_labels, rows, cols;
    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverse_int(magic);
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverse_int(magic);
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = reverse_int(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = reverse_int(num_labels);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = reverse_int(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = reverse_int(cols);
    n_cols = cols;
    n_rows = rows;

    // Seek to start_idx
    image_file.seekg(start_idx * rows * cols, std::ios::cur);
    label_file.seekg(start_idx, std::ios::cur);

    std::vector<char> pixels(rows * cols);
    char label;
    for (int i = 0; i < num_images && i < num_items; ++i) {
        image_file.read(pixels.data(), rows * cols);
        label_file.read(&label, 1);
        string sLabel = std::to_string(int(label));
        cv::Mat image_tmp(rows, cols, CV_8UC1, pixels.data());
        Image tempImage;
        tempImage.label = sLabel;
        tempImage.image = image_tmp.clone();
        container.push_back(tempImage);
    }

    image_file.close();
    label_file.close();
}

void MNIST::applyTransformation(int rotation_angle) {
    for (auto& image : train_set) {
        cv::Mat rotated;
        cv::Point2f center(image.image.cols / 2.0f, image.image.rows / 2.0f);
        cv::Mat rot_matrix = cv::getRotationMatrix2D(center, rotation_angle, 1.0);
        cv::warpAffine(image.image, rotated, rot_matrix, image.image.size());
        image.image = rotated.clone();
    }
    for (auto& image : test_set) {
        cv::Mat rotated;
        cv::Point2f center(image.image.cols / 2.0f, image.image.rows / 2.0f);
        cv::Mat rot_matrix = cv::getRotationMatrix2D(center, rotation_angle, 1.0);
        cv::warpAffine(image.image, rotated, rot_matrix, image.image.size());
        image.image = rotated.clone();
    }
}

void MNIST::run() {
    int total_train_images = 60000;
    int images_per_worker = total_train_images / (rank == 0 ? 1 : size - 1);
    int start_idx = (rank == 0) ? 0 : (rank - 1) * images_per_worker;
    int num_images = (rank == 0) ? total_train_images : (rank == size - 1) ? total_train_images - start_idx : images_per_worker;

    mnist_reader(trainImg.c_str(), trainLabel.c_str(), train_set, start_idx, num_images);
}

vector<Image>& MNIST::returnTest() {
    std::cout << test_set.size() << " test data imported." << std::endl;
    std::cout << "Image heightxwidth = " << test_set[0].image.rows << "x" << test_set[0].image.cols << std::endl;
    return test_set;
}

vector<Image>& MNIST::returnTrain() {
    std::cout << train_set.size() << " train data imported." << std::endl;
    std::cout << "Image heightxwidth = " << train_set[0].image.rows << "x" << train_set[0].image.cols << std::endl;
    return train_set;
}
