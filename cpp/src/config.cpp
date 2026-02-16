#include "config.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <algorithm>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif

namespace fs = std::filesystem;

namespace cytologick {

namespace {

// Search order for model files (same as Python version)
const std::vector<std::string> MODEL_SEARCH_ORDER = {
    "new_best.onnx",
    "model_best.onnx",
    "new_final.onnx",
    "model_final.onnx",
    "new_last.onnx",
    "model.onnx"
};

// Get directory containing the executable
fs::path getExecutableDir() {
#ifdef __APPLE__
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        return fs::path(path).parent_path();
    }
#elif defined(_WIN32)
    char path[MAX_PATH];
    if (GetModuleFileNameA(NULL, path, MAX_PATH) != 0) {
        return fs::path(path).parent_path();
    }
#elif defined(__linux__)
    char path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = '\0';
        return fs::path(path).parent_path();
    }
#endif
    return fs::current_path();
}

fs::path findConfigFile() {
    fs::path exeDir = getExecutableDir();

    // Check executable directory first (where the .exe/.app is)
    for (const auto& name : {"config.yaml", "config.yml"}) {
        fs::path path = exeDir / name;
        if (fs::exists(path)) {
            return path;
        }
    }

    // Fallback: check current working directory
    for (const auto& name : {"config.yaml", "config.yml"}) {
        fs::path path = fs::current_path() / name;
        if (fs::exists(path)) {
            return path;
        }
    }

    return {};
}

template<typename T>
T getOrDefault(const YAML::Node& node, const std::string& key, const T& defaultVal) {
    if (node[key]) {
        try {
            return node[key].as<T>();
        } catch (...) {
            return defaultVal;
        }
    }
    return defaultVal;
}

// Resolve path: if relative, make it relative to base directory
fs::path resolvePath(const fs::path& path, const fs::path& baseDir) {
    if (path.empty()) {
        return path;
    }
    if (path.is_absolute()) {
        return path;
    }
    // Relative path - resolve relative to baseDir
    return fs::weakly_canonical(baseDir / path);
}

} // anonymous namespace

Config Config::load() {
    fs::path configPath = findConfigFile();
    if (configPath.empty()) {
        std::cout << "No config file found, using defaults" << std::endl;
        // Resolve default paths relative to exe directory
        Config cfg;
        fs::path exeDir = getExecutableDir();
        cfg.slideDir = resolvePath(cfg.slideDir, exeDir);
        cfg.hddSlides = resolvePath(cfg.hddSlides, exeDir);
        return cfg;
    }
    return loadFrom(configPath);
}

Config Config::loadFrom(const fs::path& path) {
    Config cfg;

    try {
        YAML::Node root = YAML::LoadFile(path.string());

        // Image processing settings
        if (root["neural_network"]) {
            auto nn = root["neural_network"];

            if (nn["image_shape"] && nn["image_shape"].IsSequence()) {
                auto shape = nn["image_shape"];
                if (shape.size() >= 2) {
                    cfg.imageShape[0] = shape[0].as<int>();
                    cfg.imageShape[1] = shape[1].as<int>();
                }
            }

            if (nn["image_chunk"] && nn["image_chunk"].IsSequence()) {
                auto chunk = nn["image_chunk"];
                if (chunk.size() >= 2) {
                    cfg.imageChunk[0] = chunk[0].as<int>();
                    cfg.imageChunk[1] = chunk[1].as<int>();
                }
            }

            cfg.classes = getOrDefault(nn, "classes", cfg.classes);
            cfg.batchSize = getOrDefault(nn, "batch_size", cfg.batchSize);

            // Optional label list for annotation (keys of neural_network.labels mapping).
            if (nn["labels"] && nn["labels"].IsMap()) {
                cfg.annotationLabels.clear();
                for (auto it = nn["labels"].begin(); it != nn["labels"].end(); ++it) {
                    try {
                        cfg.annotationLabels.push_back(it->first.as<std::string>());
                    } catch (...) {
                        // ignore
                    }
                }
                std::sort(cfg.annotationLabels.begin(), cfg.annotationLabels.end());
                cfg.annotationLabels.erase(
                    std::unique(cfg.annotationLabels.begin(), cfg.annotationLabels.end()),
                    cfg.annotationLabels.end()
                );
            }
        }

        // General settings (paths)
        if (root["general"]) {
            auto gen = root["general"];

            if (gen["hdd_slides"]) {
                cfg.hddSlides = gen["hdd_slides"].as<std::string>();
            }
            if (gen["openslide_path"]) {
                cfg.openSlidePath = gen["openslide_path"].as<std::string>();
            }
        }

        // GUI settings (slide_dir is in gui section in Python)
        if (root["gui"]) {
            auto gui = root["gui"];

            if (gui["slide_dir"]) {
                cfg.slideDir = gui["slide_dir"].as<std::string>();
            }
            cfg.predMode = getOrDefault<std::string>(gui, "unet_pred_mode", cfg.predMode);
            cfg.defaultThreshold = getOrDefault(gui, "default_threshold", cfg.defaultThreshold);
        }

        // Model path
        if (root["model"]) {
            auto model = root["model"];
            if (model["path"]) {
                cfg.modelPath = model["path"].as<std::string>();
            }
        }

        std::cout << "Loaded config from: " << path << std::endl;

        // Resolve relative paths relative to config file directory (or exe dir)
        fs::path baseDir = path.parent_path();
        if (baseDir.empty()) {
            baseDir = getExecutableDir();
        }

        cfg.slideDir = resolvePath(cfg.slideDir, baseDir);
        cfg.hddSlides = resolvePath(cfg.hddSlides, baseDir);
        cfg.openSlidePath = resolvePath(cfg.openSlidePath, baseDir);
        cfg.modelPath = resolvePath(cfg.modelPath, baseDir);

        std::cout << "  slideDir: " << cfg.slideDir << std::endl;
        std::cout << "  hddSlides: " << cfg.hddSlides << std::endl;

    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        std::cerr << "Using default configuration" << std::endl;
    }

    return cfg;
}

fs::path Config::findModelFile() const {
    // Check explicit model path first
    if (!modelPath.empty() && fs::exists(modelPath)) {
        return modelPath;
    }

    fs::path exeDir = getExecutableDir();

    // Search in _main directory relative to executable
    std::vector<fs::path> searchDirs = {
        exeDir / "_main",           // Next to executable
        fs::current_path() / "_main" // Current working directory
    };

    for (const auto& mainDir : searchDirs) {
        if (!fs::exists(mainDir)) {
            continue;
        }

        // Try known model names
        for (const auto& name : MODEL_SEARCH_ORDER) {
            fs::path modelFile = mainDir / name;
            if (fs::exists(modelFile)) {
                return modelFile;
            }
        }

        // Fallback: find any .onnx file
        for (const auto& entry : fs::directory_iterator(mainDir)) {
            if (entry.path().extension() == ".onnx") {
                return entry.path();
            }
        }
    }

    return {};
}

} // namespace cytologick
