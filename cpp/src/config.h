#pragma once

#include <string>
#include <array>
#include <filesystem>
#include <vector>

namespace cytologick {

/**
 * Application configuration loaded from config.yaml
 * Mirrors the Python Config dataclass for compatibility
 */
struct Config {
    // Image processing
    std::array<int, 2> imageShape = {128, 128};   // Model input size
    std::array<int, 2> imageChunk = {256, 256};   // Processing chunk size
    int classes = 3;                               // Number of output classes

    // Paths (defaults match Python config.py)
    std::filesystem::path slideDir = "./current";  // SLIDE_DIR: Directory with slide files
    std::filesystem::path hddSlides = "./current"; // HDD_SLIDES: Secondary slide directory
    std::filesystem::path openSlidePath;           // OpenSlide DLL path (Windows)
    std::filesystem::path modelPath;               // Path to ONNX model

    // Inference
    int batchSize = 16;                            // Inference batch size
    float defaultThreshold = 0.6f;                 // Default confidence threshold

    // GUI
    std::string predMode = "direct";               // Prediction mode

    // Optional: label list for annotator (derived from YAML neural_network.labels keys)
    std::vector<std::string> annotationLabels;

    /**
     * Load configuration from YAML file
     * Searches: current dir -> executable dir
     * @return Loaded config with defaults for missing values
     */
    static Config load();

    /**
     * Load configuration from specific file
     * @param path Path to YAML config file
     * @return Loaded config with defaults for missing values
     */
    static Config loadFrom(const std::filesystem::path& path);

    /**
     * Find model file in _main directory
     * Searches for: new_best.pth, model.pth, etc (as .onnx)
     * @return Path to model file, or empty if not found
     */
    std::filesystem::path findModelFile() const;
};

} // namespace cytologick
