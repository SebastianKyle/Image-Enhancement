#include "HistogramTransformer.h"

std::vector<cv::Vec3f> HistogramTransformer::histogram_color_img(const cv::Mat& source_img) {
    if (!source_img.data) {
        return std::vector<cv::Vec3f>(0);
    }

    int width = source_img.cols, height = source_img.rows;

    std::vector<cv::Vec3f> histogram(256, cv::Vec3f(0, 0, 0));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                histogram[source_img.at<cv::Vec3b>(y, x)[c]][c]++;
            }
        }
    }

    for (int v = 0; v < 256; v++) {
        for (int c = 0; c < 3; c++) {
            histogram[v][c] /= (height * width);
        }
    }

    return histogram;
}

std::vector<double> HistogramTransformer::histogram_gray_img(const cv::Mat& source_img) {
    if (!source_img.data) {
        return std::vector<double>(0);
    }

    int width = source_img.cols, height = source_img.rows;

    std::vector<double> histogram(256, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_value = source_img.at<uchar>(y, x);
            histogram[pixel_value]++;
        }
    }

    for (int v = 0; v < 256; v++) {
        histogram[v] /= (height * width);
    }

    return histogram;
}

void HistogramTransformer::compute_cdf_color(const std::vector<cv::Vec3f>& histogram, std::vector<cv::Vec3f>& cdf) {
    for (int c = 0; c < 3; c++) {
        cdf[0][c] = histogram[0][c];

        for (int v = 1; v < 256; v++) {
            cdf[v][c] = cdf[v - 1][c] + histogram[v][c];
        }
    }
}

void HistogramTransformer::compute_cdf_gray(const std::vector<double>& histogram, std::vector<double>& cdf) {
    cdf[0] = histogram[0];

    for (int v = 1; v < 256; v++) {
        cdf[v] = cdf[v - 1] + histogram[v];
    }
}

int HistogramTransformer::histogram_equalize(const cv::Mat& source_img, cv::Mat& dest_img) {
    if (!source_img.data) {
        return 0;
    }

    cv::Mat output = cv::Mat::zeros(source_img.size(), source_img.type());
    int width = source_img.cols, height = source_img.rows;
    
    if (source_img.channels() == 3) {
        std::vector<cv::Vec3f> histogram = histogram_color_img(source_img); 
        std::vector<cv::Vec3f> cdf(256, cv::Vec3f(0, 0, 0));

        std::thread cdf_thread(&HistogramTransformer::compute_cdf_color, this, std::cref(histogram), std::ref(cdf));
        cdf_thread.join();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    output.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(255 * cdf[source_img.at<cv::Vec3b>(y, x)[c]][c]);
                }
            }
        }
    }
    else {
        std::vector<double> histogram = histogram_gray_img(source_img);
        std::vector<double> cdf(256, 0);

        std::thread cdf_thread(&HistogramTransformer::compute_cdf_gray, this, std::cref(histogram), std::ref(cdf));
        cdf_thread.join();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output.at<uchar>(y, x) = cv::saturate_cast<uchar>(255 * cdf[source_img.at<uchar>(y, x)]);
            }
        }
    }

    dest_img = output.clone();
    return 1;
}