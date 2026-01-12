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

cv::Mat renderOverlayFast(const cv::Mat& pathologyMap, float threshold) {
    if (pathologyMap.empty()) {
        return cv::Mat();
    }

    int height = pathologyMap.rows;
    int width = pathologyMap.cols;
    int numClasses = pathologyMap.channels();

    std::vector<cv::Mat> channels;
    cv::split(pathologyMap, channels);

    cv::Mat overlay = cv::Mat::zeros(height, width, CV_8UC4);

    cv::Mat redMask;
    cv::Mat yellowMask;

    if (numClasses > 2) {
        cv::Mat atypicalProb = channels[2];

        // class2Mask: where channel 2 is dominant
        cv::Mat class2Mask;
        {
            cv::Mat ge1, ge0;
            cv::compare(atypicalProb, channels[1], ge1, cv::CMP_GE);
            cv::compare(atypicalProb, channels[0], ge0, cv::CMP_GE);
            cv::bitwise_and(ge1, ge0, class2Mask);
        }

        // Red: above threshold
        cv::threshold(atypicalProb, redMask, threshold, 255, cv::THRESH_BINARY);
        redMask.convertTo(redMask, CV_8UC1);

        // Yellow: between lowThreshold and threshold
        float lowThreshold = 0.3f;
        if (lowThreshold < threshold) {
            cv::Mat aboveLow, belowThresh;
            cv::threshold(atypicalProb, aboveLow, lowThreshold, 255, cv::THRESH_BINARY);
            cv::threshold(atypicalProb, belowThresh, threshold, 255, cv::THRESH_BINARY_INV);
            aboveLow.convertTo(aboveLow, CV_8UC1);
            belowThresh.convertTo(belowThresh, CV_8UC1);
            cv::bitwise_and(aboveLow, belowThresh, yellowMask);
            cv::bitwise_and(yellowMask, class2Mask, yellowMask);
        }

        // Green from class2: below lowThreshold
        cv::Mat greenClass2;
        cv::threshold(atypicalProb, greenClass2, lowThreshold, 255, cv::THRESH_BINARY_INV);
        greenClass2.convertTo(greenClass2, CV_8UC1);
        cv::bitwise_and(greenClass2, class2Mask, greenClass2);

        // Apply class2 green
        overlay.setTo(cv::Scalar(0, 255, 0, 40), greenClass2);
    }

    // Process normal cells (channel 1) - green detections
    if (numClasses > 1) {
        cv::Mat normalProb = channels[1];
        cv::Mat normalMask;
        cv::threshold(normalProb, normalMask, 0.5, 255, cv::THRESH_BINARY);
        normalMask.convertTo(normalMask, CV_8UC1);

        // Exclude red and yellow areas from green
        if (!redMask.empty()) {
            cv::bitwise_and(normalMask, ~redMask, normalMask);
        }
        if (!yellowMask.empty()) {
            cv::bitwise_and(normalMask, ~yellowMask, normalMask);
        }

        // Green fill
        overlay.setTo(cv::Scalar(0, 255, 0, 64), normalMask);

        // Green outline
        auto normalContours = findContours(normalMask, 500);
        if (!normalContours.empty()) {
            cv::drawContours(overlay, normalContours, -1, cv::Scalar(0, 200, 0, 150), 1);
        }
    }

    // Apply yellow on top (alpha 127 to match full analysis)
    if (!yellowMask.empty()) {
        overlay.setTo(cv::Scalar(0, 255, 255, 127), yellowMask);
    }

    // Red outline using contours + labels
    if (!redMask.empty() && numClasses > 2) {
        cv::Mat atypicalProb = channels[2];
        auto redContours = findContours(redMask, 0);
        for (const auto& contour : redContours) {
            std::vector<std::vector<cv::Point>> cnt = {contour};
            cv::drawContours(overlay, cnt, 0, cv::Scalar(255, 0, 255, 200), 2);

            // Calculate probability and draw label
            float prob = calculateContourProbability(atypicalProb, contour);
            cv::Rect bbox = cv::boundingRect(contour);

            std::ostringstream ss;
            ss << std::fixed << std::setprecision(0) << (prob * 100) << "%";

            cv::Point textPos(
                bbox.x + bbox.width / 2 - 15,
                bbox.y + bbox.height / 2 + 5
            );

            cv::putText(overlay, ss.str(), textPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 127, 255), 2);
        }
    }

    return overlay;
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

    cv::Mat lowBinaryMask;
    cv::Mat redFillMask;

    // Process atypical cells (channel 2) with threshold
    if (numClasses > 2) {
        cv::Mat atypicalProb = channels[2];
        cv::Mat class2Mask;
        {
            cv::Mat ge1, ge0;
            cv::compare(atypicalProb, channels[1], ge1, cv::CMP_GE);
            cv::compare(atypicalProb, channels[0], ge0, cv::CMP_GE);
            cv::bitwise_and(ge1, ge0, class2Mask);
        }

        // Threshold to binary
        cv::Mat binary;
        cv::threshold(atypicalProb, binary, threshold, 255, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8UC1);

        // Find contours (no area filtering for smooth red/yellow transition)
        auto contours = findContours(binary, 0);
        cv::Mat redFill = binary.clone();
        redFillMask = redFill;

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

        // Draw sub-threshold atypical areas in yellow
        float lowThreshold = 0.3f;
        // Areas below lowThreshold -> green background
        cv::Mat lowProb;
        cv::threshold(atypicalProb, lowProb, lowThreshold, 255, cv::THRESH_BINARY_INV);
        lowProb.convertTo(lowProb, CV_8UC1);
        cv::Mat lowGreen;
        cv::bitwise_and(lowProb, class2Mask, lowGreen);
        if (!redFill.empty()) {
            cv::bitwise_and(lowGreen, ~redFill, lowGreen);
        }
        if (lowThreshold < threshold) {
            cv::Mat lowBinary;
            // Yellow zone: lowThreshold <= prob < threshold
            // Use threshold operations instead of inRange to avoid boundary issues
            cv::Mat aboveLow, belowThresh;
            cv::threshold(atypicalProb, aboveLow, lowThreshold, 255, cv::THRESH_BINARY);
            cv::threshold(atypicalProb, belowThresh, threshold, 255, cv::THRESH_BINARY_INV);
            aboveLow.convertTo(aboveLow, CV_8UC1);
            belowThresh.convertTo(belowThresh, CV_8UC1);
            cv::bitwise_and(aboveLow, belowThresh, lowBinary);
            cv::bitwise_and(lowBinary, class2Mask, lowBinary);
            // Exclude red areas from yellow
            if (!redFill.empty()) {
                cv::bitwise_and(lowBinary, ~redFill, lowBinary);
            }
            // Exclude yellow from green
            cv::bitwise_and(lowGreen, ~lowBinary, lowGreen);
            lowBinaryMask = lowBinary;
        }
        // Green fill (BGRA: B=0, G=255, R=0, A=40)
        overlay.setTo(cv::Scalar(0, 255, 0, 40), lowGreen);
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

    // Drawing order: green -> yellow -> red outline -> text

    // Create combined exclude mask (red + yellow) for green
    cv::Mat greenExcludeMask;
    if (!redFillMask.empty() && !lowBinaryMask.empty()) {
        cv::bitwise_or(redFillMask, lowBinaryMask, greenExcludeMask);
    } else if (!redFillMask.empty()) {
        greenExcludeMask = redFillMask;
    } else if (!lowBinaryMask.empty()) {
        greenExcludeMask = lowBinaryMask;
    }

    // 1. Draw green (normal) detections, excluding red and yellow areas
    drawGreenDetections(overlay, detections, greenExcludeMask);

    // 2. Apply yellow mask (already excludes red areas from earlier processing)
    if (!lowBinaryMask.empty()) {
        overlay.setTo(cv::Scalar(0, 255, 255, 127), lowBinaryMask);
    }

    // 3. Draw red outlines (atypical)
    drawRedOutlines(overlay, detections);

    // 4. Draw labels last so they appear on top of everything
    drawDetectionLabels(overlay, detections);

    return {overlay, stats};
}

void drawGreenDetections(cv::Mat& overlay, const std::vector<Detection>& detections, const cv::Mat& excludeMask) {
    // Collect all green contours
    std::vector<std::vector<cv::Point>> greenContours;
    for (const auto& det : detections) {
        if (det.classIndex == 1) {
            greenContours.push_back(det.contour);
        }
    }

    if (greenContours.empty()) {
        return;
    }

    // Create single fill mask for all green contours
    cv::Mat fillMask = cv::Mat::zeros(overlay.size(), CV_8UC1);
    cv::drawContours(fillMask, greenContours, -1, cv::Scalar(255), cv::FILLED);

    // Create single outline mask
    cv::Mat outlineMask = cv::Mat::zeros(overlay.size(), CV_8UC1);
    cv::drawContours(outlineMask, greenContours, -1, cv::Scalar(255), 1);

    // Exclude red areas
    if (!excludeMask.empty()) {
        cv::bitwise_and(fillMask, ~excludeMask, fillMask);
        cv::bitwise_and(outlineMask, ~excludeMask, outlineMask);
    }

    // Apply colors
    overlay.setTo(cv::Scalar(0, 255, 0, 64), fillMask);
    overlay.setTo(cv::Scalar(0, 200, 0, 150), outlineMask);
}

void drawRedOutlines(cv::Mat& overlay, const std::vector<Detection>& detections) {
    // Note: OpenCV uses BGRA order for cv::Scalar on 4-channel images
    for (const auto& det : detections) {
        if (det.classIndex == 2) {
            std::vector<std::vector<cv::Point>> contours = {det.contour};
            // Magenta outline (BGRA: B=255, G=0, R=255, A=200)
            cv::drawContours(overlay, contours, 0, cv::Scalar(255, 0, 255, 200), 2);
        }
    }
}

void drawDetections(cv::Mat& overlay, const std::vector<Detection>& detections, bool showLabels) {
    drawGreenDetections(overlay, detections, cv::Mat());
    drawRedOutlines(overlay, detections);
    if (showLabels) {
        drawDetectionLabels(overlay, detections);
    }
}

void drawDetectionLabels(cv::Mat& overlay, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        if (det.classIndex == 2) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(0) << (det.probability * 100) << "%";

            cv::Point textPos(
                det.boundingBox.x + det.boundingBox.width / 2 - 15,
                det.boundingBox.y + det.boundingBox.height / 2 + 5
            );

            // Magenta text matching Python style (BGRA: B=255, G=0, R=127, A=255)
            cv::putText(overlay, ss.str(), textPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 127, 255), 2);
        }
    }
}

void drawRegionBboxes(cv::Mat& overlay, const std::vector<RegionBbox>& bboxes) {
    if (overlay.empty() || bboxes.empty()) {
        return;
    }

    for (size_t idx = 0; idx < bboxes.size(); ++idx) {
        const auto& bbox = bboxes[idx];
        cv::Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);
        rect &= cv::Rect(0, 0, overlay.cols, overlay.rows);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }

        // Cyan rectangle (BGRA)
        cv::rectangle(overlay, rect, cv::Scalar(255, 255, 0, 200), 2);

        std::ostringstream label;
        label << "R" << (idx + 1) << ": "
              << std::fixed << std::setprecision(0)
              << (bbox.maxProbability * 100) << "%";

        int labelY = std::max(15, rect.y - 5);
        cv::putText(
            overlay,
            label.str(),
            cv::Point(rect.x, labelY),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 255, 0, 255),
            1,
            cv::LINE_AA
        );
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
