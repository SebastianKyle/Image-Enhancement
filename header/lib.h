#pragma once

#ifndef LIB_H
#define LIB_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <thread>
#include <tuple>

bool str_compare(const char* a, std::string b);
double char_2_double(char* argv[], int n);
int char_2_int(char* argv[], int n);

template <typename Func>
void parallel_for_(int start, int end, Func func);

void showImageWithAspectRatio(const std::string& windowName, const cv::Mat& img, int windowWidth, int windowHeight);

#endif