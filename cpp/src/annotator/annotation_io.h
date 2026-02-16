#pragma once

#include "annotation_types.h"

#include <QString>

#include <filesystem>
#include <vector>

namespace cytologick::annotation_io {

// Load annotations from JSON array [{label, points, rect}, ...].
bool loadJson(const std::filesystem::path& jsonPath,
              std::vector<Annotation>& out,
              QString* error = nullptr);

// Save annotations in Cytologick JSON format next to slide.
bool saveJson(const std::filesystem::path& jsonPath,
              const std::vector<Annotation>& annotations,
              QString* error = nullptr);

// Parse ASAP XML (basename.xml), convert to Annotation, and append only new ones.
// Returns number of appended annotations; -1 means parse/read error.
int mergeFromAsapXml(const std::filesystem::path& xmlPath,
                     std::vector<Annotation>& annotations,
                     QString* error = nullptr);

}  // namespace cytologick::annotation_io

