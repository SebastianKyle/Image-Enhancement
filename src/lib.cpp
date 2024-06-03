#include "lib.h"

bool str_compare(const char* a, std::string b) {
    if (strlen(a) != b.length()) {
        return 0;
    }

    for (int i = 0; i < strlen(a); i++) {
        if (a[i] != b[i]) return 0;
    }

    return 1;
}

double char_2_double(char* argv[], int n) {
    double temp = std::stod(argv[n], NULL);
    return temp;
}

int char_2_int(char* argv[], int n) {
    int temp = std::stoi(argv[n], NULL);
    return temp;
}

template <typename Func>
void parallel_for_(int start, int end, Func func) {
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = (end - start) / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        int chunk_start = start + i * chunk_size;
        int chunk_end = (i == num_threads - 1) ? end : chunk_start + chunk_size;
        threads.emplace_back([=]() {
            for (int j = chunk_start; j < chunk_end; ++j) {
                func(j);
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void showImageWithAspectRatio(const std::string& windowName, const cv::Mat& img, int windowWidth, int windowHeight) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, windowWidth, windowHeight);

    int imgWidth = img.cols;
    int imgHeight = img.rows;

    double imgAspect = static_cast<double>(imgWidth) / imgHeight;
    double windowAspect = static_cast<double>(windowWidth) / windowHeight;

    int displayWidth, displayHeight;
    if (windowAspect > imgAspect) {
        displayHeight = windowHeight;
        displayWidth = static_cast<int>(windowHeight * imgAspect);
    } else {
        displayWidth = windowWidth;
        displayHeight = static_cast<int>(windowWidth / imgAspect);
    }

    cv::Mat canvas(windowHeight, windowWidth, img.type(), cv::Scalar::all(0));

    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(displayWidth, displayHeight));

    int xOffset = (windowWidth - displayWidth) / 2;
    int yOffset = (windowHeight - displayHeight) / 2;
    resizedImg.copyTo(canvas(cv::Rect(xOffset, yOffset, displayWidth, displayHeight)));

    cv::imshow(windowName, canvas);
}