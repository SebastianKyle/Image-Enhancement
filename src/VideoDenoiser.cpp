#include "VideoDenoiser.h"

VideoDenoiser::VideoDenoiser() : temporalWindowSize(1)
{
    motionEstimator = new MotionEstimator();
}

VideoDenoiser::~VideoDenoiser()
{
    delete motionEstimator;
}

void VideoDenoiser::initialize(int blockSize, int pyramidLevels, int temporalWindow)
{
    this->motionEstimator->setBlockSize(blockSize);
    this->motionEstimator->setPyramidLevels(pyramidLevels);
    this->temporalWindowSize = temporalWindow;
}

cv::Mat VideoDenoiser::denoiseFrame(const std::vector<cv::Mat> &frames, int curFrameIdx)
{
    int startFrame = std::max(0, curFrameIdx - temporalWindowSize);
    int endFrame = std::min((int)frames.size() - 1, curFrameIdx + temporalWindowSize);

    cv::Mat curFrame = frames[curFrameIdx].clone();
    cv::Mat denoisedFrame = cv::Mat::zeros(curFrame.size(), CV_32FC3);
    cv::Mat weightFrame = cv::Mat::zeros(curFrame.size(), CV_32FC3);

    float sigma = temporalWindowSize / 2.5f;
    std::vector<float> temporalWeights(endFrame - startFrame + 1);

#pragma omp parallel for
    for (int i = startFrame; i <= endFrame; i++)
    {
        temporalWeights[i - startFrame] = (1 / sqrt(2 * 3.1415 * sigma * sigma)) * exp(-0.5f * pow((i - curFrameIdx) / sigma, 2));
    }

#pragma omp parallel for
    for (int i = startFrame; i <= endFrame; i++)
    {
        if (i == curFrameIdx)
            continue;

        cv::Mat motionVectors = motionEstimator->estimateMotionOF(curFrame, frames[i]);

        if (i % 20 == 0)
        {
#pragma omp critical
            {
                drawMotionVectors(curFrame, motionVectors, "Motion Vectors for Frames");
            }
        }

#pragma omp parallel for
        for (int y = 0; y < curFrame.rows; y++)
        {
            for (int x = 0; x < curFrame.cols; x++)
            {
                cv::Point2f motionVec = motionVectors.at<cv::Point2f>(y, x);

                int newY = cv::borderInterpolate(cvRound(y + motionVec.y), curFrame.rows, cv::BORDER_REFLECT_101);
                int newX = cv::borderInterpolate(cvRound(x + motionVec.x), curFrame.cols, cv::BORDER_REFLECT_101);

                if (newY >= 0 && newY < curFrame.rows && newX >= 0 && newX < curFrame.cols)
                {
                    float weight = temporalWeights[i - startFrame] / (1.0f + cv::norm(motionVec));
#pragma omp critical
                    {
                        denoisedFrame.at<cv::Vec3f>(y, x) += cv::Vec3f(frames[i].at<cv::Vec3b>(newY, newX) * weight);
                        weightFrame.at<cv::Vec3f>(y, x) += cv::Vec3f(weight, weight, weight);
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (int y = 0; y < curFrame.rows; y++)
    {
        for (int x = 0; x < curFrame.cols; x++)
        {
            cv::Vec3f weight = weightFrame.at<cv::Vec3f>(y, x);
            if (weight != cv::Vec3f(0, 0, 0))
            {
                denoisedFrame.at<cv::Vec3f>(y, x)[0] /= weight[0];
                denoisedFrame.at<cv::Vec3f>(y, x)[1] /= weight[1];
                denoisedFrame.at<cv::Vec3f>(y, x)[2] /= weight[2];
            }
            else
            {
                denoisedFrame.at<cv::Vec3f>(y, x) = cv::Vec3f(curFrame.at<cv::Vec3b>(y, x));
            }
        }
    }

    denoisedFrame.convertTo(denoisedFrame, CV_8UC3);

    return denoisedFrame;
}

void VideoDenoiser::processVideo(const std::string &inVideoPath, const std::string &outVideoPath)
{
    cv::VideoCapture cap(inVideoPath);
    if (!cap.isOpened())
    {
        throw std::runtime_error("Error opening video file: " + inVideoPath);
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    cv::VideoWriter writer(outVideoPath, fourcc, fps, cv::Size(frameWidth, frameHeight));
    if (!writer.isOpened())
    {
        throw std::runtime_error("Error opening video writer: " + outVideoPath);
    }

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty())
            break;
        frames.push_back(frame.clone());
    }

    // #pragma omp parallel for
    for (int i = 0; i < frames.size(); i++)
    {
        cv::Mat denoisedFrame = denoiseFrame(frames, i);
        writer.write(denoisedFrame);
        // #pragma omp critical
        std::cout << "\n Wrote frame " << i + 1 << ".";
    }

    cap.release();
    writer.release();
}