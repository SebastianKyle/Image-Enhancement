#pragma once
#ifndef SHARPENER_H
#define SHARPENER_H

#include "lib.h"

class Sharpener {
private:

public:
    int sharpen(const cv::Mat& source_img, cv::Mat& dest_img, double alpha);
};

#endif