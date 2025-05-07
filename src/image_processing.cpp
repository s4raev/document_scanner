#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

namespace ImageProcessing {

    cv::Mat enhanceImage(const cv::Mat& input) {
        cv::Mat enhanced;
        cv::GaussianBlur(input, enhanced, cv::Size(0, 0), 3);
        cv::addWeighted(input, 1.5, enhanced, -0.5, 0, enhanced);
        return enhanced;
    }

    std::vector<cv::Point> findDocumentContour(const cv::Mat& image) {
        cv::Mat gray, blurred, edged;
        
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1);
        cv::Canny(blurred, edged, 50, 150, 3);
        
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(edged, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1, false) > cv::contourArea(c2, false);
        });
        
        std::vector<cv::Point> approx;
        for (size_t i = 0; i < contours.size(); i++) {
            double peri = cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], approx, 0.02 * peri, true);
            
            // Если у контура 4 угла, это скорее всего документ
            if (approx.size() == 4) {
                return approx;
            }
        }
        
        std::vector<cv::Point> defaultContour;
        defaultContour.emplace_back(0, 0);
        defaultContour.emplace_back(image.cols - 1, 0);
        defaultContour.emplace_back(image.cols - 1, image.rows - 1);
        defaultContour.emplace_back(0, image.rows - 1);
        return defaultContour;
    }

    cv::Mat applyPerspectiveTransform(const cv::Mat& image, const std::vector<cv::Point>& contour) {
        std::vector<cv::Point2f> srcPoints;
        for (const auto& p : contour) {
            srcPoints.emplace_back(p.x, p.y);
        }
        
        std::vector<cv::Point2f> dstPoints;
        float width = cv::norm(srcPoints[1] - srcPoints[0]);
        float height = cv::norm(srcPoints[3] - srcPoints[0]);
        
        dstPoints.emplace_back(0, 0);
        dstPoints.emplace_back(width, 0);
        dstPoints.emplace_back(width, height);
        dstPoints.emplace_back(0, height);
        
        cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
        cv::Mat warped;
        cv::warpPerspective(image, warped, M, cv::Size(width, height));
        
        return warped;
    }

    cv::Mat applyScanEffect(const cv::Mat& image) {
        cv::Mat gray, adjusted;
        
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        cv::convertScaleAbs(gray, adjusted, 1.2, 0);
        
        cv::Mat binary;
        cv::adaptiveThreshold(adjusted, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, 11, 2);
        
        return adjusted;
    }

    cv::Mat processDocumentImage(const cv::Mat& input) {
        cv::Mat enhanced = enhanceImage(input);
        
        std::vector<cv::Point> contour = findDocumentContour(enhanced);
        
        cv::Mat warped = applyPerspectiveTransform(enhanced, contour);
        
        cv::Mat scanned = applyScanEffect(warped);
        
        return scanned;
    }
};