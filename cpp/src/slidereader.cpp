#include "slidereader.h"
#include <openslide/openslide.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

namespace cytologick {

SlideReader::SlideReader() = default;

SlideReader::~SlideReader() {
    close();
}

SlideReader::SlideReader(SlideReader&& other) noexcept
    : m_slide(other.m_slide)
    , m_path(std::move(other.m_path))
    , m_lastError(std::move(other.m_lastError))
{
    other.m_slide = nullptr;
}

SlideReader& SlideReader::operator=(SlideReader&& other) noexcept {
    if (this != &other) {
        close();
        m_slide = other.m_slide;
        m_path = std::move(other.m_path);
        m_lastError = std::move(other.m_lastError);
        other.m_slide = nullptr;
    }
    return *this;
}

bool SlideReader::open(const fs::path& path) {
    close();

    m_path = path;
    m_slide = openslide_open(path.string().c_str());

    if (!m_slide) {
        m_lastError = "Failed to open slide: " + path.string();
        return false;
    }

    if (!checkError()) {
        close();
        return false;
    }

    std::cout << "Opened slide: " << path << std::endl;
    std::cout << "Levels: " << getLevelCount() << std::endl;

    return true;
}

void SlideReader::close() {
    if (m_slide) {
        openslide_close(m_slide);
        m_slide = nullptr;
    }
    m_path.clear();
}

bool SlideReader::isOpen() const {
    return m_slide != nullptr;
}

bool SlideReader::checkError() const {
    if (!m_slide) return false;

    const char* error = openslide_get_error(m_slide);
    if (error) {
        m_lastError = error;
        std::cerr << "OpenSlide error: " << error << std::endl;
        return false;
    }
    return true;
}

int SlideReader::getLevelCount() const {
    if (!m_slide) return 0;
    return openslide_get_level_count(m_slide);
}

std::pair<int64_t, int64_t> SlideReader::getLevelDimensions(int level) const {
    if (!m_slide) return {0, 0};

    int64_t w = 0, h = 0;
    openslide_get_level_dimensions(m_slide, level, &w, &h);
    return {w, h};
}

std::vector<std::pair<int64_t, int64_t>> SlideReader::getAllLevelDimensions() const {
    std::vector<std::pair<int64_t, int64_t>> dims;
    int count = getLevelCount();

    for (int i = 0; i < count; ++i) {
        dims.push_back(getLevelDimensions(i));
    }

    return dims;
}

double SlideReader::getLevelDownsample(int level) const {
    if (!m_slide) return 1.0;
    return openslide_get_level_downsample(m_slide, level);
}

cv::Mat SlideReader::readRegion(int64_t x, int64_t y, int level, int64_t width, int64_t height) const {
    if (!m_slide || width <= 0 || height <= 0) {
        return cv::Mat();
    }

    // Allocate buffer for ARGB data (OpenSlide format)
    std::vector<uint32_t> buffer(width * height);

    openslide_read_region(m_slide, buffer.data(), x, y, level, width, height);

    if (!checkError()) {
        return cv::Mat();
    }

    // Convert from ARGB to RGBA
    // OpenSlide returns pre-multiplied ARGB in native byte order
    cv::Mat rgba(height, width, CV_8UC4);

    for (int64_t i = 0; i < height; ++i) {
        for (int64_t j = 0; j < width; ++j) {
            uint32_t pixel = buffer[i * width + j];

            // Extract ARGB components (little-endian: BGRA in memory)
            uint8_t a = (pixel >> 24) & 0xFF;
            uint8_t r = (pixel >> 16) & 0xFF;
            uint8_t g = (pixel >> 8) & 0xFF;
            uint8_t b = pixel & 0xFF;

            // Un-premultiply alpha if needed
            if (a > 0 && a < 255) {
                r = static_cast<uint8_t>(std::min(255, (r * 255) / a));
                g = static_cast<uint8_t>(std::min(255, (g * 255) / a));
                b = static_cast<uint8_t>(std::min(255, (b * 255) / a));
            }

            // Store as RGBA
            rgba.at<cv::Vec4b>(i, j) = cv::Vec4b(r, g, b, a);
        }
    }

    return rgba;
}

cv::Mat SlideReader::readRegionRGB(int64_t x, int64_t y, int level, int64_t width, int64_t height) const {
    cv::Mat rgba = readRegion(x, y, level, width, height);
    if (rgba.empty()) {
        return cv::Mat();
    }

    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
    return rgb;
}

std::vector<fs::path> SlideReader::findSlides(const fs::path& directory, bool recursive) {
    std::vector<fs::path> slides;

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return slides;
    }

    // Only MRXS files (matching Python gui.py behavior)
    const std::vector<std::string> extensions = {
        ".mrxs"
    };

    auto processEntry = [&](const fs::directory_entry& entry) {
        if (entry.is_regular_file()) {
            // Skip macOS resource fork files (start with "._")
            std::string filename = entry.path().filename().string();
            if (filename.size() >= 2 && filename[0] == '.' && filename[1] == '_') {
                return;
            }

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            for (const auto& supported : extensions) {
                if (ext == supported) {
                    slides.push_back(entry.path());
                    break;
                }
            }
        }
    };

    try {
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(directory)) {
                processEntry(entry);
            }
        } else {
            for (const auto& entry : fs::directory_iterator(directory)) {
                processEntry(entry);
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error scanning directory: " << e.what() << std::endl;
    }

    std::sort(slides.begin(), slides.end());
    return slides;
}

} // namespace cytologick
