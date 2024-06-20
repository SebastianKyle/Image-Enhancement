#include "HomomorphicFilter.h"
#include "PointWiseTransformer.h"

int HomomorphicFilter::homomorphic_filter(const cv::Mat &source_img, cv::Mat &dest_img, double gamma1, double gamma2)
{
    if (!source_img.data)
    {
        return 0;
    }

    cv::Mat output;

    cv::Mat float_img;
    source_img.convertTo(float_img, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    if (source_img.channels() == 3) {
        cv::split(float_img, channels);
    } else {
        channels.push_back(float_img);
    }

    std::vector<cv::Mat> filtered_channels(channels.size());
    for (size_t i = 0; i < channels.size(); ++i) {
        cv::Mat log_transformed;
        cv::log(channels[i] + 1, log_transformed);

        cv::Mat illumination;
        int kernel_size = 5;
        cv::GaussianBlur(log_transformed, illumination, cv::Size(kernel_size, kernel_size), 0);

        cv::Mat reflectance = log_transformed - illumination;

        cv::Mat log_output = gamma1 * illumination + gamma2 * reflectance;

        cv::Mat exp_output;
        cv::exp(log_output, exp_output);
        exp_output -= 1;

        cv::normalize(exp_output, exp_output, 0, 255, cv::NORM_MINMAX);
        exp_output.convertTo(filtered_channels[i], CV_8U);
    }

    if (source_img.channels() == 3) {
        cv::merge(filtered_channels, output);
    } else {
        output = filtered_channels[0];
    }

    dest_img = output.clone();

    return 1;
}