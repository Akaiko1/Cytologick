cmake_minimum_required(VERSION 3.20)
project(Cytologick VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(yaml-cpp REQUIRED)

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
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    yaml-cpp::yaml-cpp
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
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
