#pragma once

#include "config.h"
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

namespace cytologick {

/**
 * ONNX Runtime inference engine for U-Net segmentation model
 * Provides direct (sliding window) inference mode
 */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /**
     * Load ONNX model from file
     * @param modelPath Path to .onnx model file
     * @param useGPU Whether to use CUDA execution provider
     * @return true if model loaded successfully
     */
    bool loadModel(const std::filesystem::path& modelPath, bool useGPU = true);

    /**
     * Check if a model is loaded
     */
    bool isModelLoaded() const;

    /**
     * Run inference on an image (direct mode - sliding window)
     * @param image Input RGB image (any size)
     * @param config Configuration with tile size, classes, etc.
     * @return Probability map (H x W x Classes) with softmax applied
     */
    cv::Mat runInference(const cv::Mat& image, const Config& config);

    /**
     * Run inference using smooth windowing (overlapping tiles + spline blending).
     * @param image Input RGB image (any size)
     * @param config Configuration with tile size, classes, etc.
     * @return Probability map (H x W x Classes) with softmax applied
     */
    cv::Mat runInferenceSmooth(const cv::Mat& image, const Config& config);

    /**
     * Bounding box for atypical region detection
     */
    struct RegionBbox {
        int x, y, width, height;
        float maxProbability;
    };

    /**
     * Run inference with region-wise refinement.
     * Two-pass approach:
     * 1. Fast tiled inference to find non-background regions
     * 2. For each connected region, extract bbox, resize to model input, re-run inference
     * Only class 2 (atypical) pixels are updated from refinement pass.
     *
     * @param image Input RGB image (any size)
     * @param config Configuration with tile size, classes, etc.
     * @return Pair of (probability map, vector of atypical region bboxes)
     */
    std::pair<cv::Mat, std::vector<RegionBbox>> runInferenceRegion(const cv::Mat& image, const Config& config);

    /**
     * Get number of output classes
     */
    int getNumClasses() const { return m_numClasses; }

    /**
     * Get input shape (height, width)
     */
    std::pair<int, int> getInputShape() const { return m_inputShape; }

    /**
     * Get last error message
     */
    const std::string& getLastError() const { return m_lastError; }

private:
    // ONNX Runtime objects
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;

    // Model info
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<std::string> m_inputNameStrings;
    std::vector<std::string> m_outputNameStrings;
    std::pair<int, int> m_inputShape = {128, 128};
    int m_numClasses = 3;

    mutable std::string m_lastError;

    /**
     * Preprocess a single tile for inference
     * @param tile RGB tile (must match input shape)
     * @return Preprocessed tensor data (CHW format, normalized)
     */
    std::vector<float> preprocessTile(const cv::Mat& tile) const;

    /**
     * Run inference on a batch of tiles
     * @param tiles Vector of preprocessed tile data
     * @param batchSize Number of tiles in batch
     * @return Output tensor data with softmax applied
     */
    std::vector<float> runBatch(const std::vector<float>& tiles, int batchSize);

    /**
     * Apply softmax to logits in NCHW format
     * @param logits Raw model output in NCHW format
     * @param batchSize Number of samples in batch
     * @param numClasses Number of classes (C dimension)
     * @param height Spatial height (H dimension)
     * @param width Spatial width (W dimension)
     */
    static void applySoftmaxNCHW(std::vector<float>& logits, int batchSize, int numClasses, int height, int width);

    /**
     * Pad image to be divisible by tile size
     * @param image Input image
     * @param tileSize Tile dimensions
     * @return Padded image
     */
    static cv::Mat padImage(const cv::Mat& image, const std::pair<int, int>& tileSize);

    /**
     * Extract non-overlapping tiles from padded image
     * @param image Padded image
     * @param tileSize Tile dimensions
     * @return Vector of tiles and their top-left coordinates
     */
    static std::vector<std::pair<cv::Mat, cv::Point>> extractTiles(
        const cv::Mat& image,
        const std::pair<int, int>& tileSize
    );

    static cv::Mat padImageReflect(const cv::Mat& image, const std::pair<int, int>& tileSize, int subdivisions);
    static cv::Mat createSplineWindow2D(int height, int width, int power = 2);
};

} // namespace cytologick
