#include "PointWiseTransformer.h"
#include "HistogramTransformer.h"
#include "NoiseSmoother.h"
#include "Sharpener.h"

int main(int argc, char** argv){
    PointWiseTransformer* pointWiseTransformer = new PointWiseTransformer();
    HistogramTransformer* histogramTransformer = new HistogramTransformer(); 
    NoiseSmoother* noiseSmoother = new NoiseSmoother(); 
    Sharpener* sharpener = new Sharpener(); 

    cv::Mat dest_img;
    cv::Mat source_img = imread(argv[2], cv::IMREAD_UNCHANGED);
    if (!source_img.data) {
        std::cout << "\n Image not found (wrong path) !";
        std::cout << "\n Path: " << argv[2];
        return 0;
    }

    bool success = 0;

    if (argc == 5) {
        if (str_compare(argv[1], "-log")) {
            success = pointWiseTransformer->log_transform(source_img, dest_img, char_2_int(argv, 4));
        }
        else if (str_compare(argv[1], "-sns")) {
            success = noiseSmoother->spatially_adaptive_noise_smoothing(source_img, dest_img, char_2_int(argv, 4));
        }
        else if (str_compare(argv[1], "-median")) {
            success = noiseSmoother->median_filter(source_img, dest_img, char_2_int(argv, 4));
        }
        else if (str_compare(argv[1], "-sharp")) {
            success = sharpener->sharpen(source_img, dest_img, char_2_double(argv, 4)); 
        }

    } 
    else if (argc == 4) {
        if (str_compare(argv[1], "-constret")) {
            success = pointWiseTransformer->constrast_stretching(source_img, dest_img);
        }
        else if (str_compare(argv[1], "-hiseq")) {
            success = histogramTransformer->histogram_equalize(source_img, dest_img);
        }
    }
    else if (argc == 6) {
        if (str_compare(argv[1], "-bil")) {
            success = noiseSmoother->bilateral_filter(source_img, dest_img, char_2_int(argv, 4), char_2_int(argv, 5));
        }
    }

    if (success) {
        showImageWithAspectRatio("Source Image", source_img, 800, 600);
        showImageWithAspectRatio("Destination Image", dest_img, 800, 600);

        imwrite(argv[3], dest_img);
    }
    else {
        std::cout << "\n Something went wrong!";
    }

    cv::waitKey(0);
    return 0;
}
