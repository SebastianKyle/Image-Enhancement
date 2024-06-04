#include "Sharpener.h"

int Sharpener::sharpen(const cv::Mat& source_img, cv::Mat& dest_img, double alpha) {
    if (!source_img.data) {
        return 0;
    } 

    cv::Mat output = cv::Mat::zeros(source_img.size(), source_img.type());
    cv::Mat blurred = cv::Mat::zeros(source_img.size(), source_img.type());
    cv::GaussianBlur(source_img, blurred, cv::Size(5, 5), 0);

    cv::Mat unsharp_mask = source_img - blurred;
    output = source_img + alpha * unsharp_mask;
    
    dest_img = output.clone();
    return 1;
}