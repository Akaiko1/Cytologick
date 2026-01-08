#include "graphics.h"
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace cytologick {

std::vector<std::vector<cv::Point>> findContours(const cv::Mat& mask, int minArea) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter by minimum area
    contours.erase(
        std::remove_if(contours.begin(), contours.end(),
            [minArea](const std::vector<cv::Point>& c) {
                return cv::contourArea(c) < minArea;
            }),
        contours.end()
    );

    return contours;
}

float calculateContourProbability(const cv::Mat& probabilityChannel, const std::vector<cv::Point>& contour) {
    if (contour.empty() || probabilityChannel.empty()) {
        return 0.0f;
    }

    // Create mask for the contour
    cv::Mat mask = cv::Mat::zeros(probabilityChannel.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours = {contour};
    cv::drawContours(mask, contours, 0, cv::Scalar(255), cv::FILLED);

    // Calculate mean inside contour
    cv::Scalar mean = cv::mean(probabilityChannel, mask);
    return static_cast<float>(mean[0]);
}

std::pair<cv::Mat, DetectionStats> processDensePathologyMap(
    const cv::Mat& pathologyMap,
    float threshold,
    int minArea
) {
    DetectionStats stats;
    std::vector<Detection> detections;

    if (pathologyMap.empty()) {
        return {cv::Mat(), stats};
    }

    int height = pathologyMap.rows;
    int width = pathologyMap.cols;
    int numClasses = pathologyMap.channels();

    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(pathologyMap, channels);

    // Create RGBA overlay
    cv::Mat overlay = cv::Mat::zeros(height, width, CV_8UC4);

    // Process atypical cells (channel 2) with threshold
    if (numClasses > 2) {
        cv::Mat atypicalProb = channels[2];

        // Threshold to binary
        cv::Mat binary;
        cv::threshold(atypicalProb, binary, threshold, 255, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8UC1);

        // Find contours
        auto contours = findContours(binary, minArea);

        float totalProb = 0.0f;
        for (const auto& contour : contours) {
            Detection det;
            det.contour = contour;
            det.probability = calculateContourProbability(atypicalProb, contour);
            det.classIndex = 2;
            det.boundingBox = cv::boundingRect(contour);

            detections.push_back(det);
            stats.atypicalCells++;
            stats.maxConfidence = std::max(stats.maxConfidence, det.probability);
            totalProb += det.probability;
        }

        if (!detections.empty()) {
            stats.avgConfidence = totalProb / detections.size();
        }
    }

    // Process normal cells (channel 1) with fixed threshold
    if (numClasses > 1) {
        cv::Mat normalProb = channels[1];

        cv::Mat binary;
        cv::threshold(normalProb, binary, 0.5, 255, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8UC1);

        auto contours = findContours(binary, minArea);

        for (const auto& contour : contours) {
            Detection det;
            det.contour = contour;
            det.probability = calculateContourProbability(normalProb, contour);
            det.classIndex = 1;
            det.boundingBox = cv::boundingRect(contour);

            detections.push_back(det);
            stats.normalCells++;
        }
    }

    stats.totalDetections = stats.atypicalCells + stats.normalCells;

    // Draw detections on overlay
    drawDetections(overlay, detections, true);

    return {overlay, stats};
}

void drawDetections(cv::Mat& overlay, const std::vector<Detection>& detections, bool showLabels) {
    // Draw atypical detections (red)
    for (const auto& det : detections) {
        if (det.classIndex == 2) {
            // Red fill with transparency
            std::vector<std::vector<cv::Point>> contours = {det.contour};
            cv::drawContours(overlay, contours, 0, cv::Scalar(255, 0, 0, 127), cv::FILLED);

            // Magenta outline
            cv::drawContours(overlay, contours, 0, cv::Scalar(255, 0, 255, 200), 2);

            // Draw probability label
            if (showLabels) {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(0) << (det.probability * 100) << "%";

                cv::Point textPos(
                    det.boundingBox.x + det.boundingBox.width / 2 - 15,
                    det.boundingBox.y + det.boundingBox.height / 2 + 5
                );

                // Background for text
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(overlay,
                    cv::Point(textPos.x - 2, textPos.y - textSize.height - 2),
                    cv::Point(textPos.x + textSize.width + 2, textPos.y + baseline + 2),
                    cv::Scalar(0, 0, 0, 180),
                    cv::FILLED
                );

                cv::putText(overlay, ss.str(), textPos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 255), 1);
            }
        }
    }

    // Draw normal detections (green outline only)
    for (const auto& det : detections) {
        if (det.classIndex == 1) {
            std::vector<std::vector<cv::Point>> contours = {det.contour};
            cv::drawContours(overlay, contours, 0, cv::Scalar(0, 255, 0, 64), cv::FILLED);
            cv::drawContours(overlay, contours, 0, cv::Scalar(0, 200, 0, 150), 1);
        }
    }
}

std::string formatDetectionStats(const DetectionStats& stats) {
    std::ostringstream ss;
    ss << "Analysis Results\n";
    ss << "----------------\n";
    ss << "Total detections: " << stats.totalDetections << "\n";
    ss << "Atypical cells: " << stats.atypicalCells << "\n";
    ss << "Normal cells: " << stats.normalCells << "\n";

    if (stats.atypicalCells > 0) {
        ss << "\nMax confidence: " << std::fixed << std::setprecision(1)
           << (stats.maxConfidence * 100) << "%\n";
        ss << "Avg confidence: " << std::fixed << std::setprecision(1)
           << (stats.avgConfidence * 100) << "%";
    }

    return ss.str();
}

std::pair<int, int> getCorrectedSize(int width, int height, int factor) {
    int correctedWidth = (width / factor) * factor;
    int correctedHeight = (height / factor) * factor;
    return {correctedWidth, correctedHeight};
}

} // namespace cytologick
