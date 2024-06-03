#pragma once
#ifndef NOISESMOOTHER_H
#define NOISESMOOTHER_H

#include "lib.h"

class NoiseSmoother {
private:

public:

    /* Spatially adaptive noise smoothing */
    double compute_noise_variance_gray(const cv::Mat& source_img);
    cv::Vec3f compute_noise_variance_color(const cv::Mat& source_img);

    std::tuple<double, double> compute_local_spatial_variance_gray(const cv::Mat& source_img, int y, int x, int kernel_size);
    std::tuple<cv::Vec3f, cv::Vec3f> compute_local_spatial_variance_color(const cv::Mat& source_img, int y, int x, int kernel_size);

    int spatially_adaptive_noise_smoothing(const cv::Mat& source_img, cv::Mat& dest_img, int kernel_size);

    /* Median filter */
    int median_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);

    /* Bilateral filter */
    int bilateral_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k, int sigma_b);

    /* Alpha-trimmed mean filter */

};

#endif