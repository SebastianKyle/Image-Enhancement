#include "MotionEstimator.h"

MotionEstimator::MotionEstimator() : blockSize(16), pyramidLevels(1) {}

MotionEstimator::~MotionEstimator() {}

void MotionEstimator::setBlockSize(int size)
{
    this->blockSize = size;
}

void MotionEstimator::setPyramidLevels(int levels)
{
    this->pyramidLevels = levels;
}

cv::Mat MotionEstimator::estimateMotionBM(const cv::Mat &curFrame, const cv::Mat &prevFrame)
{
    cv::Mat curGray, prevGray;
    if (curFrame.channels() == 3)
    {
        cv::cvtColor(curFrame, curGray, cv::COLOR_RGB2GRAY);
    }
    else
    {
        curGray = curFrame;
    }

    if (prevFrame.channels() == 3)
    {
        cv::cvtColor(prevFrame, prevGray, cv::COLOR_RGB2GRAY);
    }
    else
    {
        prevGray = prevFrame;
    }

    cv::Mat motionVectors = cv::Mat::zeros(curFrame.size(), CV_32FC2);

    int blockQuarter = int((blockSize - 1) / 2);
    const int searchRange = 7;
    int height = curFrame.rows, width = curFrame.cols;

    for (int y = 0; y < height; y += this->blockSize)
    {
        for (int x = 0; x < width; x += this->blockSize)
        {
            int blockWidth = (x + blockSize > width) ? (width - x) : blockSize;
            int blockHeight = (y + blockSize > height) ? (height - y) : blockSize;

            // Define block and search region
            cv::Rect blockRect(x, y, blockWidth, blockHeight);
            cv::Mat curBlock = curGray(blockRect);

            // Logarithmic search
            int stepSize = std::max(searchRange, 1);
            cv::Point2f bestMatch(0, 0);
            double minError = std::numeric_limits<double>::max();

            while (stepSize > 0)
            {
                bool foundBetterMatch = false;

                for (int dy = -stepSize; dy <= stepSize; dy += stepSize)
                {
                    for (int dx = -stepSize; dx <= stepSize; dx += stepSize)
                    {
                        int newX = x + dx;
                        int newY = y + dy;

                        if (newX >= 0 && newX + blockWidth <= width && newY >= 0 && newY + blockHeight <= height)
                        {
                            cv::Rect candidateRect(newX, newY, blockWidth, blockHeight);
                            cv::Mat candidateBlock = prevGray(candidateRect);

                            cv::Mat diff;
                            cv::absdiff(curBlock, candidateBlock, diff);
                            double error = cv::sum(diff.mul(diff))[0];

                            // Mean absolute error
                            if (error < minError)
                            {
                                minError = error;
                                bestMatch = cv::Point2f(dx, dy);
                                foundBetterMatch = true;
                            }
                        }
                    }
                }

                stepSize /= 2;
            }

            // Found displacement vector at (y, x)
            for (int by = y; by < y + blockHeight; by++)
            {
                for (int bx = x; bx < x + blockWidth; bx++)
                {
                    motionVectors.at<cv::Point2f>(by, bx) = bestMatch;
                }
            }
        }
    }

    return motionVectors;
}

void MotionEstimator::buildPyramids(const cv::Mat &img, std::vector<cv::Mat> &pyramids, int levels)
{
    pyramids.clear();
    pyramids.push_back(img);

    for (int i = 1; i < levels; i++)
    {
        cv::Mat downsampled;
        cv::pyrDown(pyramids[i - 1], downsampled);
        pyramids.push_back(downsampled);
    }
}

cv::Mat MotionEstimator::estimateMotionOF(const cv::Mat &curFrame, const cv::Mat &prevFrame)
{
    cv::Mat curGray, prevGray;
    if (curFrame.channels() == 3)
    {
        cv::cvtColor(curFrame, curGray, cv::COLOR_RGB2GRAY);
    }
    else
    {
        curGray = curFrame;
    }

    if (prevFrame.channels() == 3)
    {
        cv::cvtColor(prevFrame, prevGray, cv::COLOR_RGB2GRAY);
    }
    else
    {
        prevGray = prevFrame;
    }

    // Image gradients
    // cv::Mat Ix, Iy, It;
    // cv::Sobel(curGray, Ix, CV_64F, 1, 0, 3);
    // cv::Sobel(curGray, Iy, CV_64F, 0, 1, 3);
    // It = curGray - prevGray;

    // cv::Mat motionVectors = cv::Mat::zeros(curFrame.size(), CV_32FC2);

    int winSize = 7; // window size for local patch
    int halfWin = winSize / 2;

    int height = curFrame.rows, width = curFrame.cols;

    // for (int y = halfWin; y < height; y++) {
    //     for (int x = halfWin; x < width; x++) {
    //         cv::Mat A = cv::Mat::zeros(2, 2, CV_64F);
    //         cv::Mat b = cv::Mat::zeros(2, 1, CV_64F);

    //         // Construct Hessian matrix A and residual vector b
    //         for (int dy = -halfWin; dy <= halfWin; dy++) {
    //             for (int dx = -halfWin; dx <= halfWin; dx++) {
    //                 double ix = Ix.at<double>(y + dy, x + dx);
    //                 double iy = Iy.at<double>(y + dy, x + dx);
    //                 double it = It.at<double>(y + dy, x + dx);

    //                 A.at<double>(0, 0) += ix * ix;
    //                 A.at<double>(0, 1) += ix * iy;
    //                 A.at<double>(1, 0) += ix * iy;
    //                 A.at<double>(1, 1) += iy * iy;

    //                 b.at<double>(0, 0) += ix * it;
    //                 b.at<double>(1, 0) += iy * it;
    //             }
    //         }

    //         // Solve the normal-equation A.u = b for the displacement vector of local patch
    //         cv::Mat u;
    //         bool invertible = cv::invert(A, u, cv::DECOMP_SVD);
    //         if (invertible) {
    //             u = -u * b;
    //         }
    //         else {

    //         }
    //         u = -A.inv() * b;

    //         motionVectors.at<cv::Point2f>(y, x) = cv::Point2f(u.at<double>(0, 0), u.at<double>(1, 0));
    //     }
    // }

    std::vector<cv::Mat> curPyr, prevPyr;
    buildPyramids(curFrame, curPyr, pyramidLevels);
    buildPyramids(prevFrame, prevPyr, pyramidLevels);

    cv::Mat motionVectors = cv::Mat::zeros(curPyr[pyramidLevels - 1].size(), CV_32FC2);

    for (int level = pyramidLevels - 1; level >= 0; --level)
    {
        const cv::Mat &curImg = curPyr[level];
        const cv::Mat &prevImg = prevPyr[level];

        if (level < pyramidLevels - 1)
        {
            cv::resize(motionVectors, motionVectors, curImg.size(), 0, 0, cv::INTER_LINEAR);
            motionVectors *= 2;
        }

        cv::Mat Ix, Iy, It;
        cv::Sobel(curImg, Ix, CV_64F, 1, 0, 3);
        cv::Sobel(curImg, Iy, CV_64F, 0, 1, 3);
        cv::subtract(curImg, prevImg, It, cv::noArray(), CV_64F);
        // It = curImg - prevImg;

#pragma omp parallel for collapse(2)
        for (int y = halfWin; y < curImg.rows - halfWin; y++)
        {
            for (int x = halfWin; x < curImg.cols - halfWin; x++)
            {
                cv::Mat A = cv::Mat::zeros(2, 2, CV_64F);
                cv::Mat b = cv::Mat::zeros(2, 1, CV_64F);

                for (int dy = -halfWin; dy <= halfWin; dy++)
                {
                    for (int dx = -halfWin; dx <= halfWin; dx++)
                    {
                        double ix = Ix.at<double>(y + dy, x + dx);
                        double iy = Iy.at<double>(y + dy, x + dx);
                        double it = It.at<double>(y + dy, x + dx);

                        A.at<double>(0, 0) += ix * ix;
                        A.at<double>(0, 1) += ix * iy;
                        A.at<double>(1, 0) += ix * iy;
                        A.at<double>(1, 1) += iy * iy;

                        b.at<double>(0, 0) += ix * it;
                        b.at<double>(1, 0) += iy * it;
                    }
                }

                cv::Mat u;
                bool invertible = cv::invert(A, u, cv::DECOMP_SVD);
                if (invertible)
                {
                    u = -u * b;
#pragma omp critical
                    motionVectors.at<cv::Point2f>(y, x) += cv::Point2f(u.at<double>(0, 0), u.at<double>(1, 0));
                }
            }
        }
    }

    return motionVectors;
}