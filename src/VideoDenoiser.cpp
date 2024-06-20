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
    cv::Mat denoisedFrame = curFrame.clone();
    // cv::Mat accumFrame = cv::Mat::zeros(curFrame.size(), curFrame.type());
    cv::Mat accumFrame = cv::Mat::zeros(curFrame.size(), CV_32FC3);
    cv::Mat countFrame = cv::Mat::zeros(curFrame.size(), CV_32S);

    for (int i = startFrame; i <= endFrame; i++)
    {
        if (i == curFrameIdx)
            continue;

        cv::Mat motionVectors = motionEstimator->estimateMotionOF(frames[i], curFrame);

        for (int y = 0; y < curFrame.rows; y++)
        {
            for (int x = 0; x < curFrame.cols; x++)
            {
                cv::Point2f motionVec = motionVectors.at<cv::Point2f>(y, x);
                int newY = cv::borderInterpolate(cvRound(y + motionVec.y), curFrame.rows, cv::BORDER_REFLECT_101);
                int newX = cv::borderInterpolate(cvRound(x + motionVec.x), curFrame.cols, cv::BORDER_REFLECT_101);

                if (newY >= 0 && newY < curFrame.rows && newX >= 0 && newX < curFrame.cols)
                {
                    accumFrame.at<cv::Vec3f>(y, x) += frames[i].at<cv::Vec3b>(newY, newX);
                    countFrame.at<int>(y, x)++;
                }
            }
        }
    }

    for (int y = 0; y < curFrame.rows; y++)
    {
        for (int x = 0; x < curFrame.cols; x++)
        {
            if (countFrame.at<int>(y, x) > 0)
            {
                denoisedFrame.at<cv::Vec3b>(y, x) = accumFrame.at<cv::Vec3f>(y, x) / countFrame.at<int>(y, x);
            }
        }
    }

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
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    // int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');

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

    for (int i = 0; i < frames.size(); i++)
    {
        cv::Mat denoisedFrame = denoiseFrame(frames, i);
        writer.write(denoisedFrame);
    }

    cap.release();
    writer.release();
}