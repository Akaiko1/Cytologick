#pragma once

#include <filesystem>

namespace cytologick {

// Load the last user-selected working directory for slide discovery.
std::filesystem::path loadRememberedWorkingDir();

// Persist the working directory for future launches.
void saveRememberedWorkingDir(const std::filesystem::path& dir);

} // namespace cytologick

