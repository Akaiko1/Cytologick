#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <map>
#include <string>

namespace cytologick {

/**
 * Detection result for a single contour
 */
struct Detection {
    std::vector<cv::Point> contour;  // Contour points
    float probability;                // Average probability inside contour
    int classIndex;                   // Class index (0=background, 1=normal, 2=atypical)
    cv::Rect boundingBox;            // Bounding rectangle
};

/**
 * Statistics from pathology map processing
 */
struct DetectionStats {
    int totalDetections = 0;
    int atypicalCells = 0;
    int normalCells = 0;
    float maxConfidence = 0.0f;
    float avgConfidence = 0.0f;
};

/**
 * Process a dense probability map into visualization overlay
 *
 * @param pathologyMap Probability map from inference (H x W x Classes)
 * @param threshold Confidence threshold for atypical detection (default 0.6)
 * @param minArea Minimum contour area in pixels (default 500)
 * @return Pair of (RGBA overlay image, detection statistics)
 */
std::pair<cv::Mat, DetectionStats> processDensePathologyMap(
    const cv::Mat& pathologyMap,
    float threshold = 0.6f,
    int minArea = 500
);

/**
 * Find contours in a binary mask
 *
 * @param mask Binary mask (CV_8UC1)
 * @param minArea Minimum contour area
 * @return Vector of contours
 */
std::vector<std::vector<cv::Point>> findContours(
    const cv::Mat& mask,
    int minArea = 500
);

/**
 * Calculate average probability inside a contour
 *
 * @param probabilityChannel Single channel probability map
 * @param contour Contour points
 * @return Average probability (0-1)
 */
float calculateContourProbability(
    const cv::Mat& probabilityChannel,
    const std::vector<cv::Point>& contour
);

/**
 * Draw detection overlay with labels
 *
 * @param overlay Output RGBA image to draw on
 * @param detections List of detections to draw
 * @param showLabels Whether to draw probability labels
 */
void drawDetections(
    cv::Mat& overlay,
    const std::vector<Detection>& detections,
    bool showLabels = true
);

/**
 * Format detection statistics as display string
 *
 * @param stats Detection statistics
 * @return Formatted string for display
 */
std::string formatDetectionStats(const DetectionStats& stats);

/**
 * Correct size to be divisible by a factor
 *
 * @param width Input width
 * @param height Input height
 * @param factor Factor to align to
 * @return Pair of (corrected_width, corrected_height)
 */
std::pair<int, int> getCorrectedSize(int width, int height, int factor);

} // namespace cytologick
