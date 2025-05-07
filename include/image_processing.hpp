#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace ImageProcessing {
    cv::Mat enhanceImage(const cv::Mat& input);

    std::vector<cv::Point> findDocumentContour(const cv::Mat& image);

    cv::Mat applyPerspectiveTransform(const cv::Mat& image, const std::vector<cv::Point>& contour);

    cv::Mat applyScanEffect(const cv::Mat& image);

    cv::Mat processDocumentImage(const cv::Mat& input);
};