#include "inference.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace cytologick {

InferenceEngine::InferenceEngine() {
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Cytologick");
    m_memoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
}

InferenceEngine::~InferenceEngine() = default;

bool InferenceEngine::loadModel(const std::filesystem::path& modelPath, bool useGPU) {
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Try to use CUDA if available and requested
        if (useGPU) {
            try {
                OrtCUDAProviderOptions cudaOptions;
                cudaOptions.device_id = 0;
                sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
                std::cout << "Using CUDA execution provider" << std::endl;
            } catch (const Ort::Exception& e) {
                std::cout << "CUDA not available, using CPU: " << e.what() << std::endl;
            }
        }

        // Load the model
#ifdef _WIN32
        m_session = std::make_unique<Ort::Session>(*m_env, modelPath.wstring().c_str(), sessionOptions);
#else
        m_session = std::make_unique<Ort::Session>(*m_env, modelPath.string().c_str(), sessionOptions);
#endif

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;

        size_t numInputs = m_session->GetInputCount();
        m_inputNameStrings.clear();
        m_inputNames.clear();
        for (size_t i = 0; i < numInputs; ++i) {
            auto name = m_session->GetInputNameAllocated(i, allocator);
            m_inputNameStrings.push_back(name.get());
            m_inputNames.push_back(m_inputNameStrings.back().c_str());
        }

        size_t numOutputs = m_session->GetOutputCount();
        m_outputNameStrings.clear();
        m_outputNames.clear();
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = m_session->GetOutputNameAllocated(i, allocator);
            m_outputNameStrings.push_back(name.get());
            m_outputNames.push_back(m_outputNameStrings.back().c_str());
        }

        // Get input shape
        auto inputShape = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() >= 4) {
            // Assuming NCHW format
            m_inputShape.first = static_cast<int>(inputShape[2]);   // Height
            m_inputShape.second = static_cast<int>(inputShape[3]);  // Width
        }

        // Get output shape to determine number of classes
        auto outputShape = m_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() >= 2) {
            m_numClasses = static_cast<int>(outputShape[1]);  // Classes dimension
        }

        std::cout << "Model loaded: " << modelPath << std::endl;
        std::cout << "Input shape: " << m_inputShape.first << "x" << m_inputShape.second << std::endl;
        std::cout << "Output classes: " << m_numClasses << std::endl;

        return true;

    } catch (const Ort::Exception& e) {
        m_lastError = std::string("ONNX Runtime error: ") + e.what();
        std::cerr << m_lastError << std::endl;
        m_session.reset();
        return false;
    }
}

bool InferenceEngine::isModelLoaded() const {
    return m_session != nullptr;
}

std::vector<float> InferenceEngine::preprocessTile(const cv::Mat& tile) const {
    cv::Mat resized;
    if (tile.rows != m_inputShape.first || tile.cols != m_inputShape.second) {
        cv::resize(tile, resized, cv::Size(m_inputShape.second, m_inputShape.first));
    } else {
        resized = tile;
    }

    // Convert to float and normalize to [0, 1]
    cv::Mat floatMat;
    resized.convertTo(floatMat, CV_32F, 1.0 / 255.0);

    // Convert HWC to CHW format
    std::vector<float> tensorData(3 * m_inputShape.first * m_inputShape.second);
    int hw = m_inputShape.first * m_inputShape.second;

    for (int h = 0; h < m_inputShape.first; ++h) {
        for (int w = 0; w < m_inputShape.second; ++w) {
            cv::Vec3f pixel = floatMat.at<cv::Vec3f>(h, w);
            int idx = h * m_inputShape.second + w;
            tensorData[0 * hw + idx] = pixel[0];  // R
            tensorData[1 * hw + idx] = pixel[1];  // G
            tensorData[2 * hw + idx] = pixel[2];  // B
        }
    }

    return tensorData;
}

void InferenceEngine::applySoftmaxNCHW(std::vector<float>& logits, int batchSize, int numClasses, int height, int width) {
    // Data is in NCHW format: [batch, classes, height, width]
    int hw = height * width;
    int chw = numClasses * hw;

    for (int b = 0; b < batchSize; ++b) {
        for (int p = 0; p < hw; ++p) {
            // For each spatial position, apply softmax across channels
            // Find max for numerical stability
            float maxVal = logits[b * chw + 0 * hw + p];
            for (int c = 1; c < numClasses; ++c) {
                maxVal = std::max(maxVal, logits[b * chw + c * hw + p]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int c = 0; c < numClasses; ++c) {
                int idx = b * chw + c * hw + p;
                logits[idx] = std::exp(logits[idx] - maxVal);
                sum += logits[idx];
            }

            // Normalize
            for (int c = 0; c < numClasses; ++c) {
                int idx = b * chw + c * hw + p;
                logits[idx] /= sum;
            }
        }
    }
}

std::vector<float> InferenceEngine::runBatch(const std::vector<float>& tiles, int batchSize) {
    if (!m_session) {
        return {};
    }

    int tileSize = 3 * m_inputShape.first * m_inputShape.second;
    std::vector<int64_t> inputShape = {
        batchSize,
        3,
        m_inputShape.first,
        m_inputShape.second
    };

    try {
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *m_memoryInfo,
            const_cast<float*>(tiles.data()),
            tiles.size(),
            inputShape.data(),
            inputShape.size()
        );

        auto outputs = m_session->Run(
            Ort::RunOptions{nullptr},
            m_inputNames.data(),
            &inputTensor,
            1,
            m_outputNames.data(),
            m_outputNames.size()
        );

        // Get output data
        float* outputData = outputs[0].GetTensorMutableData<float>();
        auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t outputSize = 1;
        for (auto dim : outputShape) {
            outputSize *= dim;
        }

        std::vector<float> result(outputData, outputData + outputSize);

        // Output shape is NCHW: [batch, classes, height, width]
        int outH = (outputShape.size() > 2) ? static_cast<int>(outputShape[2]) : m_inputShape.first;
        int outW = (outputShape.size() > 3) ? static_cast<int>(outputShape[3]) : m_inputShape.second;

        // Apply softmax in NCHW format
        applySoftmaxNCHW(result, batchSize, m_numClasses, outH, outW);

        return result;

    } catch (const Ort::Exception& e) {
        m_lastError = std::string("Inference error: ") + e.what();
        std::cerr << m_lastError << std::endl;
        return {};
    }
}

cv::Mat InferenceEngine::padImage(const cv::Mat& image, const std::pair<int, int>& tileSize) {
    int padH = (tileSize.first - (image.rows % tileSize.first)) % tileSize.first;
    int padW = (tileSize.second - (image.cols % tileSize.second)) % tileSize.second;

    if (padH == 0 && padW == 0) {
        return image.clone();
    }

    cv::Mat padded;
    cv::copyMakeBorder(image, padded, 0, padH, 0, padW, cv::BORDER_REPLICATE);
    return padded;
}

std::vector<std::pair<cv::Mat, cv::Point>> InferenceEngine::extractTiles(
    const cv::Mat& image,
    const std::pair<int, int>& tileSize
) {
    std::vector<std::pair<cv::Mat, cv::Point>> tiles;

    int tileH = tileSize.first;
    int tileW = tileSize.second;

    for (int y = 0; y < image.rows; y += tileH) {
        for (int x = 0; x < image.cols; x += tileW) {
            cv::Rect roi(x, y, tileW, tileH);
            tiles.emplace_back(image(roi).clone(), cv::Point(x, y));
        }
    }

    return tiles;
}

cv::Mat InferenceEngine::runInference(const cv::Mat& image, const Config& config) {
    if (!m_session) {
        m_lastError = "No model loaded";
        return cv::Mat();
    }

    if (image.empty()) {
        m_lastError = "Empty input image";
        return cv::Mat();
    }

    std::cout << "Input image: " << image.cols << "x" << image.rows
              << " channels: " << image.channels() << std::endl;

    // Ensure RGB format
    cv::Mat rgb;
    if (image.channels() == 4) {
        cv::cvtColor(image, rgb, cv::COLOR_RGBA2RGB);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 3) {
        rgb = image.clone();
    } else {
        m_lastError = "Unsupported image format";
        return cv::Mat();
    }

    // Pad image
    cv::Mat padded = padImage(rgb, m_inputShape);
    std::cout << "Padded image: " << padded.cols << "x" << padded.rows << std::endl;

    // Extract tiles
    auto tiles = extractTiles(padded, m_inputShape);
    int numTiles = static_cast<int>(tiles.size());
    std::cout << "Extracted " << numTiles << " tiles of size "
              << m_inputShape.second << "x" << m_inputShape.first << std::endl;

    // Create output probability map
    cv::Mat output = cv::Mat::zeros(padded.rows, padded.cols, CV_32FC(m_numClasses));

    // Process in batches
    int batchSize = config.batchSize;
    int tileSize = 3 * m_inputShape.first * m_inputShape.second;
    int outputTileSize = m_numClasses * m_inputShape.first * m_inputShape.second;

    float maxProb0 = 0, maxProb1 = 0, maxProb2 = 0;

    for (int start = 0; start < numTiles; start += batchSize) {
        int end = std::min(start + batchSize, numTiles);
        int currentBatch = end - start;

        // Preprocess tiles
        std::vector<float> batchData;
        batchData.reserve(currentBatch * tileSize);

        for (int i = start; i < end; ++i) {
            auto tileData = preprocessTile(tiles[i].first);
            batchData.insert(batchData.end(), tileData.begin(), tileData.end());
        }

        // Run inference
        auto batchOutput = runBatch(batchData, currentBatch);

        if (batchOutput.empty()) {
            return cv::Mat();
        }

        // Place results in output map
        for (int i = 0; i < currentBatch; ++i) {
            int tileIdx = start + i;
            cv::Point pos = tiles[tileIdx].second;

            // Extract this tile's output (NCHW format -> HWC for output)
            for (int h = 0; h < m_inputShape.first; ++h) {
                for (int w = 0; w < m_inputShape.second; ++w) {
                    float* outPixel = output.ptr<float>(pos.y + h, pos.x + w);
                    for (int c = 0; c < m_numClasses; ++c) {
                        // Output is in NCHW format: [batch, classes, height, width]
                        int srcIdx = i * outputTileSize + c * m_inputShape.first * m_inputShape.second + h * m_inputShape.second + w;
                        float val = batchOutput[srcIdx];
                        outPixel[c] = val;

                        // Track max probabilities for debug
                        if (c == 0) maxProb0 = std::max(maxProb0, val);
                        else if (c == 1) maxProb1 = std::max(maxProb1, val);
                        else if (c == 2) maxProb2 = std::max(maxProb2, val);
                    }
                }
            }
        }
    }

    std::cout << "Max probabilities - class0: " << maxProb0
              << ", class1: " << maxProb1 << ", class2: " << maxProb2 << std::endl;

    // Crop to original size
    cv::Rect originalSize(0, 0, image.cols, image.rows);
    return output(originalSize).clone();
}

} // namespace cytologick
