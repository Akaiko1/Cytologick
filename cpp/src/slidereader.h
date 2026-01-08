#pragma once

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <opencv2/core.hpp>

// Forward declare OpenSlide handle
typedef struct _openslide openslide_t;

namespace cytologick {

/**
 * Wrapper for OpenSlide library to read whole-slide images
 * Supports MRXS and other formats
 */
class SlideReader {
public:
    SlideReader();
    ~SlideReader();

    // Non-copyable
    SlideReader(const SlideReader&) = delete;
    SlideReader& operator=(const SlideReader&) = delete;

    // Movable
    SlideReader(SlideReader&& other) noexcept;
    SlideReader& operator=(SlideReader&& other) noexcept;

    /**
     * Open a slide file
     * @param path Path to slide file (MRXS, SVS, etc.)
     * @return true if successfully opened
     */
    bool open(const std::filesystem::path& path);

    /**
     * Close the current slide
     */
    void close();

    /**
     * Check if a slide is currently open
     */
    bool isOpen() const;

    /**
     * Get the number of zoom levels available
     */
    int getLevelCount() const;

    /**
     * Get dimensions at a specific zoom level
     * @param level Zoom level (0 = highest resolution)
     * @return Pair of (width, height)
     */
    std::pair<int64_t, int64_t> getLevelDimensions(int level) const;

    /**
     * Get all level dimensions
     * @return Vector of (width, height) pairs for each level
     */
    std::vector<std::pair<int64_t, int64_t>> getAllLevelDimensions() const;

    /**
     * Get the downsample factor for a level
     * @param level Zoom level
     * @return Downsample factor relative to level 0
     */
    double getLevelDownsample(int level) const;

    /**
     * Read a region from the slide
     * @param x X coordinate at level 0
     * @param y Y coordinate at level 0
     * @param level Zoom level to read from
     * @param width Width of region at the specified level
     * @param height Height of region at the specified level
     * @return RGBA image as cv::Mat (4 channels)
     */
    cv::Mat readRegion(int64_t x, int64_t y, int level, int64_t width, int64_t height) const;

    /**
     * Read a region and convert to RGB
     * @param x X coordinate at level 0
     * @param y Y coordinate at level 0
     * @param level Zoom level to read from
     * @param width Width of region at the specified level
     * @param height Height of region at the specified level
     * @return RGB image as cv::Mat (3 channels)
     */
    cv::Mat readRegionRGB(int64_t x, int64_t y, int level, int64_t width, int64_t height) const;

    /**
     * Get the path of the currently open slide
     */
    const std::filesystem::path& getPath() const { return m_path; }

    /**
     * Get last error message
     */
    const std::string& getLastError() const { return m_lastError; }

    /**
     * Find all slide files in a directory
     * @param directory Directory to search
     * @param recursive Whether to search subdirectories
     * @return List of slide file paths
     */
    static std::vector<std::filesystem::path> findSlides(
        const std::filesystem::path& directory,
        bool recursive = true
    );

private:
    openslide_t* m_slide = nullptr;
    std::filesystem::path m_path;
    mutable std::string m_lastError;

    bool checkError() const;
};

} // namespace cytologick
