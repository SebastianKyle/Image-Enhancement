#pragma once
#ifndef VIDEODENOISER_H
#define VIDEODENOISER_H

#include "lib.h"
#include "MotionEstimator.h"

class VideoDenoiser
{
private:
    MotionEstimator *motionEstimator;
    int temporalWindowSize;

public:
    VideoDenoiser();
    ~VideoDenoiser();

    void initialize(int blockSize, int pyramidLevels, int temporalWindow);
    cv::Mat denoiseFrame(const std::vector<cv::Mat> &frames, int curFrameIdx);
    void processVideo(const std::string &inVideoPath, const std::string &outVideoPath);
};

#endif