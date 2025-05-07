#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "image_processing.hpp"

class DocumentScanner {
public:
    DocumentScanner();
    ~DocumentScanner() = default;
    
    // Загрузить изображение из файла
    bool loadImage(const std::string& filePath);
    
    // Обработать изображение
    void processImage();
    
    // Сохранить результат
    bool saveResult(const std::string& filePath);
    
    // Получить исходное изображение
    cv::Mat getOriginalImage() const { return originalImage_; }
    
    // Получить обработанное изображение
    cv::Mat getProcessedImage() const { return processedImage_; }
    
private:
    cv::Mat originalImage_;
    cv::Mat processedImage_;
};