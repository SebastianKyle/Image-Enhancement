#include "PointWiseTransformer.h"

int PointWiseTransformer::log_transform(const cv::Mat &source_img, cv::Mat &dest_img)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;
    bool isColor = source_img.channels() == 3;
    double c = 0;

    double max_val;
    if (isColor)
    {
        cv::Mat planes[3];
        cv::split(source_img, planes);

        for (int i = 0; i < 3; i++)
        {
            cv::minMaxLoc(planes[i], nullptr, &max_val, nullptr, nullptr);
            c = 255 / log(1 + max_val);
            planes[i].convertTo(planes[i], CV_32F);
            cv::log(planes[i] + 1, planes[i]);
            planes[i] *= c;
            planes[i].convertTo(planes[i], CV_8U);
        }
        cv::merge(planes, 3, output);
    }
    else
    {
        cv::minMaxLoc(source_img, nullptr, &max_val, nullptr, nullptr);
        c = 255 / log(1 + max_val);
        source_img.convertTo(output, CV_32F);
        cv::log(output + 1, output);
        output *= c;
        output.convertTo(output, CV_8U);
    }

    dest_img = output.clone();
    return 1;
}

int PointWiseTransformer::constrast_stretching(const cv::Mat &source_img, cv::Mat &dest_img)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = cv::Mat::zeros(source_img.size(), source_img.type());
    int width = source_img.cols, height = source_img.rows;

    uchar max_gray_pixel = 0, min_gray_pixel = 255;

    if (source_img.channels() == 3)
    {
        std::vector<cv::Mat> channels;
        cv::split(source_img, channels);

        for (int i = 0; i < 3; i++)
        {
            double minVal, maxVal;
            cv::minMaxLoc(channels[i], &minVal, &maxVal);

            if (maxVal > max_gray_pixel)
                max_gray_pixel = static_cast<uchar>(maxVal);
            if (minVal < min_gray_pixel)
                min_gray_pixel = static_cast<uchar>(minVal);
        }
    }
    else
    { // For grayscale images
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                uchar pixel_value = source_img.at<uchar>(y, x);

                if (pixel_value > max_gray_pixel)
                {
                    max_gray_pixel = pixel_value;
                }

                if (pixel_value < min_gray_pixel)
                {
                    min_gray_pixel = pixel_value;
                }
            }
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (source_img.channels() == 3)
            {
                for (int i = 0; i < 3; i++)
                {
                    output.at<cv::Vec3b>(y, x)[i] = cv::saturate_cast<uchar>((255.0 / (max_gray_pixel - min_gray_pixel)) * (source_img.at<cv::Vec3b>(y, x)[i] - min_gray_pixel));
                }
            }
            else
            {
                output.at<uchar>(y, x) = cv::saturate_cast<uchar>((255.0 / (max_gray_pixel - min_gray_pixel)) * (source_img.at<uchar>(y, x) - min_gray_pixel));
            }
        }
    }

    dest_img = output.clone();
    return 1;
}
