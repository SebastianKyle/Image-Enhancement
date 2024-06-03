#include "PointWiseTransformer.h"

int PointWiseTransformer::log_transform(const cv::Mat& source_img, cv::Mat& dest_img, double c) {
    if (!source_img.data) 
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;
    bool isColor = source_img.channels() == 3;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (source_img.channels() == 3) {
                cv::Vec3b log_transformed(0, 0, 0);

                log_transformed[0] = cv::saturate_cast<uchar>(c * log(source_img.at<cv::Vec3b>(y, x)[0] + 1));
                log_transformed[1] = cv::saturate_cast<uchar>(c * log(source_img.at<cv::Vec3b>(y, x)[1] + 1));
                log_transformed[2] = cv::saturate_cast<uchar>(c * log(source_img.at<cv::Vec3b>(y, x)[2] + 1));

                output.at<cv::Vec3b>(y, x) = log_transformed;
            }
            else {
                int log_transformed = c * log(source_img.at<uchar>(y, x) + 1);

                output.at<uchar>(y, x) = cv::saturate_cast<uchar>(log_transformed);
            }
        }
    }

    // cv::parallel_for_(cv::Range(0, height * width), [&](const Range& range) {
    //     for (int r = range.start; r < range.end; r++) {
    //         int y = r / width;
    //         int x = r % width;

    //         if (isColor) {
    //             cv::Vec3b& pixel = source_img.at<cv::Vec3b>(y, x);
    //             cv::Vec3b log_transformed(0, 0, 0);

    //             for (int i = 0; i < 3; i++) {
    //                 log_transformed[i] = cv::saturate_cast<uchar>(c * log(pixel[i] + 1));
    //             }

    //             output.at<cv::Vec3b>(y, x) = log_transformed;
    //         }
    //         else {
    //             uchar& pixel = source_img.at<uchar>(y, x);
                
    //             output.at<uchar>(y, x) = cv::saturate_cast<uchar>(c * log(pixel + 1));
    //         }
    //     }
    // });

    dest_img = output.clone();
    return 1;
}

int PointWiseTransformer::constrast_stretching(const cv::Mat& source_img, cv::Mat& dest_img) {
    if (!source_img.data) 
        return 0;

    cv::Mat output = cv::Mat::zeros(source_img.size(), source_img.type());
    int width = source_img.cols, height = source_img.rows;

    uchar max_gray_pixel = 0, min_gray_pixel = 255;

    if (source_img.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(source_img, channels);

        for (int i = 0; i < 3; i++) {
            double minVal, maxVal;
            cv::minMaxLoc(channels[i], &minVal, &maxVal);
            
            if (maxVal > max_gray_pixel)
                max_gray_pixel = static_cast<uchar>(maxVal);
            if (minVal < min_gray_pixel)
                min_gray_pixel = static_cast<uchar>(minVal);
        }
    } else { // For grayscale images
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar pixel_value = source_img.at<uchar>(y, x);

                if (pixel_value > max_gray_pixel) {
                    max_gray_pixel = pixel_value;
                }

                if (pixel_value < min_gray_pixel) {
                    min_gray_pixel = pixel_value;
                }
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (source_img.channels() == 3) {
                for (int i = 0; i < 3; i++) {
                    output.at<cv::Vec3b>(y, x)[i] = cv::saturate_cast<uchar>((255.0 / (max_gray_pixel - min_gray_pixel)) * (source_img.at<cv::Vec3b>(y, x)[i] - min_gray_pixel));
                }
            } else {
                output.at<uchar>(y, x) = cv::saturate_cast<uchar>((255.0 / (max_gray_pixel - min_gray_pixel)) * (source_img.at<uchar>(y, x) - min_gray_pixel));
            }
        }
    }

    dest_img = output.clone();
    return 1;
}
