#pragma once

#ifndef MOTIONESTIMATOR_H
#define MOTIONESTIMATOR_H

#include "lib.h"

class MotionEstimator
{
private:
    int blockSize;
    int pyramidLevels;

public:
    MotionEstimator();
    ~MotionEstimator();

    void setBlockSize(int size);
    void setPyramidLevels(int levels);

    bool isFlatRegion(const cv::Mat& source_img, const cv::Rect& rect);

    /* Block Matching */
    cv::Mat estimateMotionBM(const cv::Mat &curFrame, const cv::Mat &prevFrame);

    /* Optical Flow */
    void buildPyramids(const cv::Mat &img, std::vector<cv::Mat> &pyramids, int levels);
    cv::Mat estimateMotionOF(const cv::Mat &curFrame, const cv::Mat &prevFrame);
};

void drawMotionVectors(const cv::Mat &frame, const cv::Mat &motionVectors, const std::string& windowName = "Motion Vectors");

#endif