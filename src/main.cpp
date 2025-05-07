#include <opencv2/opencv.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

#define GL_SILENCE_DEPRECATION

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;

struct Texture {
    GLuint id = 0;
    int width = 0;
    int height = 0;
    bool dirty = true;
};

Mat originalImage;
Mat processedImage;
Texture originalTexture;
Texture processedTexture;
bool imageLoaded = false;
bool processingDone = false;
char filePath[256] = "";

bool compareContourAreas(vector<Point> contour1, vector<Point> contour2) {
    return contourArea(contour1) > contourArea(contour2);
}

bool compareXCords(Point p1, Point p2) { return p1.x < p2.x; }
bool compareYCords(Point p1, Point p2) { return p1.y < p2.y; }

double _distance(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

void resizeToHeight(Mat src, Mat &dst, int height) {
    Size s = Size(src.cols * (height / double(src.rows)), height);
    resize(src, dst, s, INTER_AREA);
}

void orderPoints(vector<Point> inpts, vector<Point> &ordered) {
    sort(inpts.begin(), inpts.end(), compareXCords);
    vector<Point> lm(inpts.begin(), inpts.begin() + 2);
    vector<Point> rm(inpts.end() - 2, inpts.end());

    sort(lm.begin(), lm.end(), compareYCords);
    Point tl(lm[0]), bl(lm[1]);

    vector<pair<Point, Point>> tmp;
    for (size_t i = 0; i < rm.size(); i++) {
        tmp.push_back(make_pair(tl, rm[i]));
    }

    sort(tmp.begin(), tmp.end(), [](auto &a, auto &b) {
        return _distance(a.first, a.second) < _distance(b.first, b.second);
    });

    ordered = {tl, tmp[0].second, tmp[1].second, bl};
}

void fourPointTransform(Mat src, Mat &dst, vector<Point> pts) {
    vector<Point> ordered_pts;
    orderPoints(pts, ordered_pts);

    double wa = _distance(ordered_pts[2], ordered_pts[3]);
    double wb = _distance(ordered_pts[1], ordered_pts[0]);
    double mw = max(wa, wb);

    double ha = _distance(ordered_pts[1], ordered_pts[2]);
    double hb = _distance(ordered_pts[0], ordered_pts[3]);
    double mh = max(ha, hb);

    Point2f src_[] = {
        Point2f(ordered_pts[0].x, ordered_pts[0].y),
        Point2f(ordered_pts[1].x, ordered_pts[1].y),
        Point2f(ordered_pts[2].x, ordered_pts[2].y),
        Point2f(ordered_pts[3].x, ordered_pts[3].y)};
    Point2f dst_[] = {
        Point2f(0, 0),
        Point2f(mw - 1, 0),
        Point2f(mw - 1, mh - 1),
        Point2f(0, mh - 1)};
    Mat m = getPerspectiveTransform(src_, dst_);
    warpPerspective(src, dst, m, Size(mw, mh));
}

void preProcess(Mat src, Mat &dst) {
    Mat gray, open, closed, blurred;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
    morphologyEx(gray, open, MORPH_OPEN, kernel);
    morphologyEx(open, closed, MORPH_CLOSE, kernel);
    GaussianBlur(closed, blurred, Size(7, 7), 0);
    Canny(blurred, dst, 75, 100);
}

void UpdateTexture(const Mat& image, Texture& texture) {
    if (image.empty()) return;

    if (texture.id == 0) {
        glGenTextures(1, &texture.id);
    }

    glBindTexture(GL_TEXTURE_2D, texture.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    Mat imageToUpload;
    if (image.channels() == 1) {
        // Для grayscale изображения конвертируем в RGB
        cvtColor(image, imageToUpload, COLOR_GRAY2RGB);
    } else {
        // Для цветного изображения оставляем как есть
        imageToUpload = image.clone();
        cvtColor(imageToUpload, imageToUpload, COLOR_BGR2RGB);
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageToUpload.cols, imageToUpload.rows, 
                0, GL_RGB, GL_UNSIGNED_BYTE, imageToUpload.data);

    texture.width = imageToUpload.cols;
    texture.height = imageToUpload.rows;
    texture.dirty = false;
}

void ProcessDocument() {
    if (!imageLoaded) return;

    Mat image = originalImage.clone();
    Mat orig = image.clone();
    double ratio = image.rows / 500.0;
    resizeToHeight(image, image, 500);

    Mat edged;
    preProcess(image, edged);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edged, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> approx(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        double peri = arcLength(contours[i], true);
        approxPolyDP(contours[i], approx[i], 0.02 * peri, true);
    }

    sort(approx.begin(), approx.end(), compareContourAreas);

    Mat warped;
    for (size_t i = 0; i < approx.size(); i++) {
        if (approx[i].size() == 4) {
            vector<Point> scaled_pts;
            for (auto& p : approx[i]) {
                scaled_pts.push_back(Point(p.x * ratio, p.y * ratio));
            }
            fourPointTransform(orig, warped, scaled_pts);
            break;
        }
    }

    if (!warped.empty()) {
        cvtColor(warped, warped, COLOR_BGR2GRAY);
        adaptiveThreshold(warped, warped, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 15);
        GaussianBlur(warped, warped, Size(3, 3), 0);
        
        processedImage = warped.clone();
        UpdateTexture(processedImage, processedTexture);
        
        processingDone = true;
        cout << "Document processed successfully" << endl;
    } else {
        cout << "Failed to find document contour" << endl;
    }
}

void ShowImage(const Texture& texture) {
    if (texture.id == 0) return;
    
    ImVec2 availSize = ImGui::GetContentRegionAvail();
    float scale = min(availSize.x / texture.width, availSize.y / texture.height);
    ImVec2 displaySize(texture.width * scale, texture.height * scale);
    
    ImVec2 pos = ImVec2(
        (availSize.x - displaySize.x) * 0.5f,
        (availSize.y - displaySize.y) * 0.5f
    );
    ImGui::SetCursorPos(pos);
    ImGui::Image((ImTextureID)(intptr_t)texture.id, displaySize);
}

int main(int, char**) {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, 
                                        "Document Scanner", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_HEIGHT));
        
        ImGui::Begin("Document Scanner", nullptr, 
                    ImGuiWindowFlags_NoResize | 
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoCollapse);

        ImGui::BeginChild("Controls", ImVec2(WINDOW_WIDTH * 0.3f, 0), true);
        
        ImGui::Text("Image Path:");
        ImGui::InputText("##filepath", filePath, IM_ARRAYSIZE(filePath));
        
        if (ImGui::Button("Load Image", ImVec2(-1, 0))) {
            if (strlen(filePath) > 0) {
                originalImage = imread(filePath);
                if (!originalImage.empty()) {
                    imageLoaded = true;
                    processingDone = false;
                    UpdateTexture(originalImage, originalTexture);
                    cout << "Image loaded: " << filePath << endl;
                }
            }
        }
        
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        
        if (ImGui::Button("Process Document", ImVec2(-1, 0))) {
            if (imageLoaded) {
                ProcessDocument();
            }
        }
        
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        
        if (ImGui::Button("Save Result", ImVec2(-1, 0)) && processingDone) {
            string savePath = fs::path(filePath).stem().string() + "_scanned.jpg";
            if (imwrite(savePath, processedImage)) {
                cout << "Saved to: " << savePath << endl;
            } else {
                cout << "Failed to save image" << endl;
            }
        }
        
        ImGui::EndChild();
        
        ImGui::SameLine();
        ImGui::BeginChild("Image View", ImVec2(0, 0), true);
        
        if (imageLoaded) {
            if (processingDone) {
                ShowImage(processedTexture);
            } else {
                ShowImage(originalTexture);
            }
        } else {
            ImVec2 textSize = ImGui::CalcTextSize("Load an image to begin");
            ImVec2 pos = ImVec2(
                (ImGui::GetContentRegionAvail().x - textSize.x) * 0.5f,
                (ImGui::GetContentRegionAvail().y - textSize.y) * 0.5f
            );
            ImGui::SetCursorPos(pos);
            ImGui::Text("Load an image to begin");
        }
        
        ImGui::EndChild();
        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (originalTexture.id != 0) glDeleteTextures(1, &originalTexture.id);
    if (processedTexture.id != 0) glDeleteTextures(1, &processedTexture.id);
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}