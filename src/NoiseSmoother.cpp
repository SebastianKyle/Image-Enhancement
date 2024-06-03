#include "NoiseSmoother.h"

double NoiseSmoother::compute_noise_variance_gray(const cv::Mat& source_img) {
    if (!source_img.data) {
        return 0;
    }

    double noise_variance = 0;
    int width = source_img.cols, height = source_img.rows;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double convolve = 0;

            convolve = source_img.at<uchar>(y - 1, x - 1)
                    - 2 * source_img.at<uchar>(y - 1, x)
                    +     source_img.at<uchar>(y - 1, x + 1)
                    - 2 * source_img.at<uchar>(y, x - 1)
                    + 4 * source_img.at<uchar>(y, x)
                    - 2 * source_img.at<uchar>(y, x + 1)
                    +     source_img.at<uchar>(y + 1, x - 1)
                    - 2 * source_img.at<uchar>(y + 1, x)
                    +     source_img.at<uchar>(y + 1, x + 1);
            
            noise_variance += convolve * convolve;
        }
    } 

    noise_variance = (noise_variance * sqrt(3.1415 / 2)) / (6 * (width - 2) * (height - 2));

    return noise_variance;
}

cv::Vec3f NoiseSmoother::compute_noise_variance_color(const cv::Mat& source_img) {
    if (!source_img.data) {
        return 0;
    }

    cv::Vec3f noise_variance(0, 0, 0);
    int width = source_img.cols, height = source_img.rows;

    if (source_img.channels() == 3) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                cv::Vec3i convolve(0, 0, 0);

                for (int c = 0; c < 3; c++) {
                    convolve[c] = source_img.at<cv::Vec3b>(y - 1, x - 1)[c]
                            - 2 * source_img.at<cv::Vec3b>(y - 1, x)[c]
                            +     source_img.at<cv::Vec3b>(y - 1, x + 1)[c]
                            - 2 * source_img.at<cv::Vec3b>(y, x - 1)[c]
                            + 4 * source_img.at<cv::Vec3b>(y, x)[c]
                            - 2 * source_img.at<cv::Vec3b>(y, x + 1)[c]
                            +     source_img.at<cv::Vec3b>(y + 1, x - 1)[c]
                            - 2 * source_img.at<cv::Vec3b>(y + 1, x)[c]
                            +     source_img.at<cv::Vec3b>(y + 1, x + 1)[c];
                    
                    noise_variance[c] += convolve[c] * convolve[c];
                }
            }
        } 

        for (int c = 0; c < 3; c++) {     
            noise_variance[c] = (noise_variance[c] * sqrt(3.1415 / 2)) / (6 * (width - 2) * (height - 2));
        }
    }

    return noise_variance;
}

std::tuple<double, double> NoiseSmoother::compute_local_spatial_variance_gray(const cv::Mat& source_img, int y, int x, int kernel_size) {
    if (!source_img.data) {
        return std::tuple<double, double>();
    }

    int quarter_side = int((kernel_size - 1) / 2);
    int height = source_img.rows, width = source_img.cols;

    double avg_pixel_value = 0;

    for (int yi = y - quarter_side; yi <= y + quarter_side; yi++) {
        for (int xi = x - quarter_side; xi <= x + quarter_side; xi++) {
            int y_coord = std::min(height - 1, abs(yi));
            int x_coord = std::min(width - 1, abs(xi));

            avg_pixel_value += source_img.at<uchar>(y_coord, x_coord);
        }
    }

    avg_pixel_value /= kernel_size * kernel_size;

    double spatial_variance = 0;
    for (int yi = y - quarter_side; yi <= y + quarter_side; yi++) {
        for (int xi = x - quarter_side; xi <= x + quarter_side; xi++) {
            int y_coord = std::min(height - 1, abs(yi));
            int x_coord = std::min(width - 1, abs(xi));

            spatial_variance += (source_img.at<uchar>(y_coord, x_coord) - avg_pixel_value) * (source_img.at<uchar>(y_coord, x_coord) - avg_pixel_value);
        }
    }

    spatial_variance /= kernel_size * kernel_size;

    std::tuple<double, double> result = std::make_tuple(spatial_variance, avg_pixel_value);

    return result;
}

std::tuple<cv::Vec3f, cv::Vec3f> NoiseSmoother::compute_local_spatial_variance_color(const cv::Mat& source_img, int y, int x, int kernel_size) {
    if (!source_img.data) {
        return std::tuple<cv::Vec3f, cv::Vec3f>();
    }

    int quarter_side = int((kernel_size - 1) / 2);
    int height = source_img.rows, width = source_img.cols;

    cv::Vec3f avg_pixel_value(0, 0, 0);

    for (int yi = y - quarter_side; yi <= y + quarter_side; yi++) {
        for (int xi = x - quarter_side; xi <= x + quarter_side; xi++) {
            int y_coord = std::min(height - 1, abs(yi));
            int x_coord = std::min(width - 1, abs(xi));

            for (int c = 0; c < 3; c++) {
                avg_pixel_value[c] += source_img.at<cv::Vec3b>(y_coord, x_coord)[c];
            }
        }
    }

    for (int c = 0; c < 3; c++) {
        avg_pixel_value[c] /= kernel_size * kernel_size;
    }

    cv::Vec3f spatial_variance(0, 0, 0);
    for (int yi = y - quarter_side; yi <= y + quarter_side; yi++) {
        for (int xi = x - quarter_side; xi <= x + quarter_side; xi++) {
            int y_coord = std::min(height - 1, abs(yi));
            int x_coord = std::min(width - 1, abs(xi));

            for (int c = 0; c < 3; c++) {
                spatial_variance[c] += (source_img.at<cv::Vec3b>(y_coord, x_coord)[c] - avg_pixel_value[c]) * (source_img.at<cv::Vec3b>(y_coord, x_coord)[c] - avg_pixel_value[c]);
            }
        }
    }

    spatial_variance /= kernel_size * kernel_size;

    std::tuple<cv::Vec3f, cv::Vec3f> result = std::make_tuple(spatial_variance, avg_pixel_value);

    return result;
}

int NoiseSmoother::spatially_adaptive_noise_smoothing(const cv::Mat& source_img, cv::Mat& dest_img, int kernel_size) {
    if (!source_img.data) {
        return 0;
    }

    int width = source_img.cols, height = source_img.rows;
    cv::Mat output = cv::Mat::zeros(source_img.size(), source_img.type());

    if (source_img.channels() == 3) {
        cv::Vec3f noise_variance = compute_noise_variance_color(source_img);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::tuple<cv::Vec3f, cv::Vec3f> res = compute_local_spatial_variance_color(source_img, y, x, kernel_size);
                cv::Vec3f spatial_variance = std::get<0>(res);
                cv::Vec3f avg_pixel_value = std::get<1>(res);

                for (int c = 0; c < 3; c++) {
                    double weight = noise_variance[c] / (spatial_variance[c] + noise_variance[c]);
                    output.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>((1 - weight) * source_img.at<cv::Vec3b>(y, x)[c]
                        + weight * avg_pixel_value[c]);
                }
            }
        }
    }
    else {
        double noise_variance = compute_noise_variance_gray(source_img);
  
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::tuple<double, double> res = compute_local_spatial_variance_gray(source_img, y, x, kernel_size);
                double spatial_variance = std::get<0>(res);
                double avg_pixel_value = std::get<1>(res);

                double weight = noise_variance / (spatial_variance + noise_variance);
                output.at<uchar>(y, x) = cv::saturate_cast<uchar>((1 - weight) * source_img.at<uchar>(y, x)
                    + weight * avg_pixel_value);
            }
        }
    }

    dest_img = output.clone();
    return 1;
}

int NoiseSmoother::median_filter(const cv::Mat &source_img, cv::Mat &dest_img, int k)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int quarter_side = int((k - 1) / 2);

            if (source_img.channels() == 3)
            {
                std::vector<int> medians[3];

                for (int yi = y - quarter_side; yi <= y + quarter_side; yi++)
                {
                    for (int xi = x - quarter_side; xi <= x + quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(yi));
                        int x_coord = std::min(width - 1, abs(xi));

                        cv::Vec3b pixel = source_img.at<cv::Vec3b>(y_coord, x_coord);
                        for (int c = 0; c < 3; c++)
                        {
                            medians[c].push_back(pixel[c]);
                        }
                    }
                }

                cv::Vec3b median;
                for (int c = 0; c < 3; c++)
                {
                    std::sort(medians[c].begin(), medians[c].end());
                    median[c] = medians[c][medians[c].size() / 2];
                }

                output.at<cv::Vec3b>(y, x) = median;
            }
            else
            {
                std::vector<int> medians;

                for (int yi = y - quarter_side; yi <= y + quarter_side; yi++)
                {
                    for (int xi = x - quarter_side; xi <= x + quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(yi));
                        int x_coord = std::min(width - 1, abs(xi));

                        medians.push_back(source_img.at<uchar>(y_coord, x_coord));
                    }
                }

                std::sort(medians.begin(), medians.end());
                uchar median = medians[medians.size() / 2];

                output.at<uchar>(y, x) = median;
            }
        }
    }

    dest_img = output.clone();
    return 1;
}

int NoiseSmoother::bilateral_filter(const cv::Mat &source_img, cv::Mat &dest_img, int k, int sigma_b)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int quarter_side = int((k - 1) / 2);
            int sigma = int(k / 6);
            sigma = sigma == 0 ? 1 : sigma;

            if (source_img.channels() == 3)
            {
                cv::Vec3i convolve(0, 0, 0);
                cv::Vec3d wsb(0, 0, 0);

                for (int yi = -quarter_side; yi <= quarter_side; yi++)
                {
                    for (int xi = -quarter_side; xi <= quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(y + yi));
                        int x_coord = std::min(width - 1, abs(x + xi));

                        cv::Vec3b pixel = source_img.at<cv::Vec3b>(y_coord, x_coord);
                        for (int c = 0; c < 3; c++)
                        {
                            // Spatial gaussian
                            double distance = yi * yi + xi * xi;
                            double n_sigma_s = exp(-distance / (2 * sigma * sigma));

                            // Brightness gaussian
                            double brightness_diff = (pixel[c] - source_img.at<cv::Vec3b>(y, x)[c]) * (pixel[c] - source_img.at<cv::Vec3b>(y, x)[c]);
                            double n_sigma_b = exp(-brightness_diff / (2 * sigma_b * sigma_b));

                            convolve[c] += pixel[c] * n_sigma_s * n_sigma_b;
                            wsb[c] += n_sigma_s * n_sigma_b;
                        }
                    }
                }

                cv::Vec3b convolve_b;
                convolve_b[0] = cv::saturate_cast<uchar>(convolve[0] / wsb[0]);
                convolve_b[1] = cv::saturate_cast<uchar>(convolve[1] / wsb[1]);
                convolve_b[2] = cv::saturate_cast<uchar>(convolve[2] / wsb[2]);

                output.at<cv::Vec3b>(y, x) = convolve_b;
            }
            else
            {
                int convolve = 0;
                double wsb = 0;

                for (int yi = -quarter_side; yi <= quarter_side; yi++)
                {
                    for (int xi = -quarter_side; xi <= quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(y + yi));
                        int x_coord = std::min(width - 1, abs(x + xi));

                        // Spatial gaussian
                        double distance_sq = yi * yi + xi * xi;
                        double n_sigma_s = exp(-distance_sq / (2 * 3.14 * sigma * sigma));

                        // Brightness gaussian
                        double brightness_diff_sq = (source_img.at<uchar>(y_coord, x_coord) - source_img.at<uchar>(y, x)) * (source_img.at<uchar>(y_coord, x_coord) - source_img.at<uchar>(y, x));
                        double n_sigma_b = exp(-brightness_diff_sq / (2 * sigma_b * sigma_b));

                        convolve += source_img.at<uchar>(y_coord, x_coord) * n_sigma_s * n_sigma_b;
                        wsb += n_sigma_s * n_sigma_b;
                    }
                }
                convolve = int(convolve / wsb);

                output.at<uchar>(y, x) = cv::saturate_cast<uchar>(convolve);
            }
        }
    }

    dest_img = output.clone();
    return 1;
}