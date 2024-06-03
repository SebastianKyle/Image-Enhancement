#pragma once
#ifndef POINTWISETRANSFORMER_H
#define POINTWISETRANSFORMER_H

#include "lib.h"

class PointWiseTransformer {
private:

public:
    int log_transform(const cv::Mat& source_img, cv::Mat& dest_img, double c);
    int constrast_stretching(const cv::Mat& source_img, cv::Mat& dest_img);
};

#endif