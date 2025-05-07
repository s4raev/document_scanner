#include "document_scanner.hpp"

DocumentScanner::DocumentScanner() {}

bool DocumentScanner::loadImage(const std::string& filePath) {
    originalImage_ = cv::imread(filePath);
    return !originalImage_.empty();
}

void DocumentScanner::processImage() {
    if (!originalImage_.empty()) {
        processedImage_ = ImageProcessing::processDocumentImage(originalImage_);
    }
}

bool DocumentScanner::saveResult(const std::string& filePath) {
    if (processedImage_.empty()) {
        return false;
    }
    return cv::imwrite(filePath, processedImage_);
}