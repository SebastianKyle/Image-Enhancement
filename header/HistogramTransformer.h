#pragma once
#ifndef HISTOGRAMTRANSFORMER_H
#define HISTOGRAMTRANSFORMER_H

#include "lib.h"

class HistogramTransformer {
private:

public:
    std::vector<cv::Vec3f> histogram_color_img(const cv::Mat& source_img); 
    std::vector<double> histogram_gray_img(const cv::Mat& source_img);
    void compute_cdf_color(const std::vector<cv::Vec3f>& histogram, std::vector<cv::Vec3f>& cdf);
    void compute_cdf_gray(const std::vector<double>& histogram, std::vector<double>& cdf);

    int histogram_equalize(const cv::Mat& source_img, cv::Mat& dest_img);
};

#endif