#pragma once
#ifndef HOMOMORPHICFILTER_H
#define HOMOMORPHICFILTER_H

#include "lib.h"

class HomomorphicFilter {
private:

public:
    int homomorphic_filter(const cv::Mat& source_img, cv::Mat& dest_img, double gamma1, double gamma2); 
};

#endif