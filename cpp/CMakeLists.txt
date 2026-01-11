cmake_minimum_required(VERSION 3.20)
project(Cytologick VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Concurrent)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Normalize yaml-cpp target name across distros.
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
else()
    # Fallback for module mode; use library variable if no target was exported.
    set(YAML_CPP_TARGET ${YAML_CPP_LIBRARIES})
endif()

# ONNX Runtime - try find_package first, fall back to manual paths
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    # Manual path configuration for Windows/CI builds
    if(onnxruntime_INCLUDE_DIR AND onnxruntime_LIB)
        message(STATUS "Using manual ONNX Runtime paths")
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${onnxruntime_LIB}"
            IMPORTED_IMPLIB "${onnxruntime_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Install via package manager or set onnxruntime_INCLUDE_DIR and onnxruntime_LIB.")
    endif()
endif()

# OpenSlide - may need manual path on Windows
find_path(OPENSLIDE_INCLUDE_DIR openslide/openslide.h
    HINTS
        $ENV{OPENSLIDE_DIR}/include
        "C:/openslide/include"
        "/usr/include"
        "/usr/local/include"
)
find_library(OPENSLIDE_LIBRARY openslide
    HINTS
        $ENV{OPENSLIDE_DIR}/lib
        "C:/openslide/lib"
        "/usr/lib"
        "/usr/local/lib"
)

if(NOT OPENSLIDE_INCLUDE_DIR OR NOT OPENSLIDE_LIBRARY)
    message(FATAL_ERROR "OpenSlide not found. Set OPENSLIDE_DIR environment variable.")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/config.cpp
    src/slidereader.cpp
    src/inference.cpp
    src/graphics.cpp
    src/mainwindow.cpp
    src/menuwindow.cpp
    src/previewwindow.cpp
)

set(HEADERS
    src/config.h
    src/slidereader.h
    src/inference.h
    src/graphics.h
    src/mainwindow.h
    src/menuwindow.h
    src/previewwindow.h
)

# Resources
set(RESOURCES
    resources/cytologick.qrc
)

# Create executable
add_executable(${PROJECT_NAME} WIN32
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENSLIDE_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Concurrent
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    ${YAML_CPP_TARGET}
    ${OPENSLIDE_LIBRARY}
)

# Windows-specific settings
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WIN32_EXECUTABLE TRUE
    )

    # Copy Qt DLLs to output directory
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Qt6::Core>
            $<TARGET_FILE:Qt6::Gui>
            $<TARGET_FILE:Qt6::Widgets>
            $<TARGET_FILE:Qt6::Concurrent>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
