#include "PointWiseTransformer.h"
#include "HistogramTransformer.h"
#include "NoiseSmoother.h"
#include "Sharpener.h"
#include "HomomorphicFilter.h"
#include "VideoDenoiser.h"

int main(int argc, char **argv)
{
    PointWiseTransformer *pointWiseTransformer = new PointWiseTransformer();
    HistogramTransformer *histogramTransformer = new HistogramTransformer();
    NoiseSmoother *noiseSmoother = new NoiseSmoother();
    Sharpener *sharpener = new Sharpener();
    HomomorphicFilter *homomorphicFilter = new HomomorphicFilter();
    VideoDenoiser *videoDenoiser = new VideoDenoiser();

    bool success = 0;

    if (str_compare(argv[1], "-image"))
    {
        cv::Mat dest_img;
        cv::Mat source_img = imread(argv[3], cv::IMREAD_UNCHANGED);
        if (!source_img.data)
        {
            std::cout << "\n Image not found (wrong path) !";
            std::cout << "\n Path: " << argv[3];
            return 0;
        }

        if (str_compare(argv[2], "-log"))
        {
            success = pointWiseTransformer->log_transform(source_img, dest_img);
        }
        else if (str_compare(argv[2], "-constret"))
        {
            success = pointWiseTransformer->constrast_stretching(source_img, dest_img);
        }
        else if (str_compare(argv[2], "-sns"))
        {
            success = noiseSmoother->spatially_adaptive_noise_smoothing(source_img, dest_img, char_2_int(argv, 5));
        }
        else if (str_compare(argv[2], "-median"))
        {
            success = noiseSmoother->median_filter(source_img, dest_img, char_2_int(argv, 5));
        }
        else if (str_compare(argv[2], "-bil"))
        {
            success = noiseSmoother->bilateral_filter(source_img, dest_img, char_2_int(argv, 5), char_2_int(argv, 6));
        }
        else if (str_compare(argv[2], "-sharp"))
        {
            success = sharpener->sharpen(source_img, dest_img, char_2_double(argv, 5));
        }
        else if (str_compare(argv[2], "-hiseq"))
        {
            success = histogramTransformer->histogram_equalize(source_img, dest_img);
        }
        else if (str_compare(argv[2], "-homo"))
        {
            success = homomorphicFilter->homomorphic_filter(source_img, dest_img, char_2_double(argv, 5), char_2_double(argv, 6));
        }

        if (success)
        {
            showImageWithAspectRatio("Source Image", source_img, 800, 600);
            showImageWithAspectRatio("Destination Image", dest_img, 800, 600);

            imwrite(argv[4], dest_img);
        }
        else
        {
            std::cout << "\n Something went wrong!";
        }
    }
    else if (str_compare(argv[1], "-video"))
    {
        if (str_compare(argv[2], "-denoise"))
        {
            if (str_compare(argv[5], "-bm"))
            {
                videoDenoiser->initialize(char_2_int(argv, 6), 1, char_2_int(argv, 7));
            }
            else if (str_compare(argv[5], "-of"))
            {
                videoDenoiser->initialize(3, char_2_int(argv, 6), char_2_int(argv, 7));
            }

            try
            {
                videoDenoiser->processVideo(argv[3], argv[4]);
            }
            catch (const std::runtime_error &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
    }

    cv::waitKey(0);
    return 0;
}

// -image

// -video -denoise "in path" "out path" -bm "9" "2" -bil
// -video -denoise "in path" "out path" -of "8" "2" -bil