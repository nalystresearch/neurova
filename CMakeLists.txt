# Neurova - OpenCV-compatible Computer Vision Library
# CMake Build System
cmake_minimum_required(VERSION 3.18)

project(Neurova 
    VERSION 0.1.0
    DESCRIPTION "High-performance computer vision library with OpenCV compatibility"
    LANGUAGES C CXX
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Options
option(NEUROVA_BUILD_SHARED_LIBS "Build shared libraries" ON)
option(NEUROVA_BUILD_TESTS "Build tests" ON)
option(NEUROVA_BUILD_EXAMPLES "Build examples" ON)
option(NEUROVA_BUILD_DOCS "Build documentation" OFF)
option(NEUROVA_BUILD_PYTHON "Build Python bindings" ON)

# Hardware acceleration options
option(NEUROVA_WITH_CUDA "Build with CUDA support" OFF)
option(NEUROVA_WITH_OPENCL "Build with OpenCL support" OFF)
option(NEUROVA_WITH_METAL "Build with Metal support" OFF)
option(NEUROVA_WITH_VULKAN "Build with Vulkan compute support" OFF)

# Math/parallel runtime options
option(NEUROVA_WITH_TBB "Build with Intel TBB" OFF)
option(NEUROVA_WITH_OPENMP "Build with OpenMP" ON)
option(NEUROVA_WITH_OPENBLAS "Build with OpenBLAS" OFF)
option(NEUROVA_WITH_MKL "Build with Intel MKL" OFF)
option(NEUROVA_WITH_IPP "Build with Intel IPP" OFF)
option(NEUROVA_WITH_LAPACK "Build with LAPACK" OFF)

# Codec options
option(NEUROVA_WITH_JPEG "Build with libjpeg-turbo" ON)
option(NEUROVA_WITH_PNG "Build with libpng" ON)
option(NEUROVA_WITH_TIFF "Build with libtiff" OFF)
option(NEUROVA_WITH_WEBP "Build with libwebp" OFF)
option(NEUROVA_WITH_OPENJPEG "Build with OpenJPEG (JPEG2000)" OFF)
option(NEUROVA_WITH_OPENEXR "Build with OpenEXR" OFF)
option(NEUROVA_WITH_AVIF "Build with libavif" OFF)
option(NEUROVA_WITH_JASPER "Build with JasPer" OFF)

# Video/capture options
option(NEUROVA_WITH_FFMPEG "Build with FFmpeg/libav" OFF)
option(NEUROVA_WITH_V4L2 "Build with V4L2 (Linux)" OFF)
option(NEUROVA_WITH_MSMF "Build with Media Foundation (Windows)" OFF)
option(NEUROVA_WITH_AVFOUNDATION "Build with AVFoundation (macOS)" OFF)

# GUI options
option(NEUROVA_WITH_QT "Build with Qt support" OFF)
option(NEUROVA_WITH_GTK "Build with GTK support" OFF)
option(NEUROVA_WITH_OPENGL "Build with OpenGL support" OFF)

# DNN options
option(NEUROVA_WITH_ONNX "Build with ONNX Runtime" OFF)
option(NEUROVA_WITH_PROTOBUF "Build with Protobuf" OFF)
option(NEUROVA_WITH_FLATBUFFERS "Build with FlatBuffers" OFF)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Platform detection
include(cmake/NeurovaUtils.cmake)
include(cmake/NeurovaCompilerFlags.cmake)

# Find dependencies
include(cmake/NeurovaDependencies.cmake)

# Hardware acceleration
if(NEUROVA_WITH_CUDA)
    include(cmake/FindNeurovaCUDA.cmake)
endif()

if(NEUROVA_WITH_OPENCL)
    include(cmake/FindNeurovaOpenCL.cmake)
endif()

if(NEUROVA_WITH_METAL)
    include(cmake/FindNeurovaMetal.cmake)
endif()

if(NEUROVA_WITH_VULKAN)
    include(cmake/FindNeurovaVulkan.cmake)
endif()

# Math libraries
if(NEUROVA_WITH_TBB)
    include(cmake/FindNeurovaTBB.cmake)
endif()

if(NEUROVA_WITH_MKL)
    include(cmake/FindNeurovaMKL.cmake)
endif()

if(NEUROVA_WITH_IPP)
    include(cmake/FindNeurovaIPP.cmake)
endif()

if(NEUROVA_WITH_OPENBLAS)
    include(cmake/FindNeurovaOpenBLAS.cmake)
endif()

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/neurova_version.h.in"
    "${CMAKE_BINARY_DIR}/include/neurova/version.h"
)

# Core library sources
set(NEUROVA_CORE_SOURCES
    src/core/matrix.cpp
    src/core/array.cpp
    src/core/memory.cpp
    src/core/parallel.cpp
    src/core/utility.cpp
)

set(NEUROVA_IMGPROC_SOURCES
    src/imgproc/filters.cpp
    src/imgproc/transform.cpp
    src/imgproc/color.cpp
    src/imgproc/morphology.cpp
    src/imgproc/histogram.cpp
    src/imgproc/contours.cpp
    src/imgproc/drawing.cpp
)

set(NEUROVA_IO_SOURCES
    src/io/image_io.cpp
    src/io/video_io.cpp
    src/io/camera.cpp
)

set(NEUROVA_HIGHGUI_SOURCES
    src/highgui/window.cpp
    src/highgui/trackbar.cpp
    src/highgui/events.cpp
)

set(NEUROVA_FEATURES_SOURCES
    src/features/detectors.cpp
    src/features/descriptors.cpp
    src/features/matchers.cpp
)

set(NEUROVA_CALIB_SOURCES
    src/calib3d/calibration.cpp
    src/calib3d/stereo.cpp
    src/calib3d/pose.cpp
)

set(NEUROVA_DNN_SOURCES
    src/dnn/net.cpp
    src/dnn/layers.cpp
    src/dnn/inference.cpp
    src/dnn/onnx_importer.cpp
)

# Combine all sources
set(NEUROVA_SOURCES
    ${NEUROVA_CORE_SOURCES}
    ${NEUROVA_IMGPROC_SOURCES}
    ${NEUROVA_IO_SOURCES}
    ${NEUROVA_HIGHGUI_SOURCES}
    ${NEUROVA_FEATURES_SOURCES}
    ${NEUROVA_CALIB_SOURCES}
    ${NEUROVA_DNN_SOURCES}
)

# Add GPU sources if enabled
if(NEUROVA_WITH_CUDA)
    enable_language(CUDA)
    list(APPEND NEUROVA_SOURCES
        src/cuda/cuda_core.cu
        src/cuda/cuda_imgproc.cu
        src/cuda/cuda_filters.cu
        src/cuda/cuda_dnn.cu
    )
endif()

if(NEUROVA_WITH_OPENCL)
    list(APPEND NEUROVA_SOURCES
        src/opencl/opencl_core.cpp
        src/opencl/opencl_imgproc.cpp
    )
endif()

if(NEUROVA_WITH_METAL)
    list(APPEND NEUROVA_SOURCES
        src/metal/metal_core.mm
        src/metal/metal_imgproc.mm
    )
endif()

# Create library
if(NEUROVA_BUILD_SHARED_LIBS)
    add_library(neurova SHARED ${NEUROVA_SOURCES})
else()
    add_library(neurova STATIC ${NEUROVA_SOURCES})
endif()

# Include directories
target_include_directories(neurova
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(neurova
    PUBLIC
        ${NEUROVA_EXTERNAL_LIBS}
    PRIVATE
        ${NEUROVA_PRIVATE_LIBS}
)

# Set library properties
set_target_properties(neurova PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/neurova.hpp"
)

# Platform-specific sources
if(WIN32)
    target_sources(neurova PRIVATE
        src/platform/windows/capture_msmf.cpp
        src/platform/windows/display_win32.cpp
    )
    target_link_libraries(neurova PRIVATE mfplat mfreadwrite mfuuid)
elseif(APPLE)
    target_sources(neurova PRIVATE
        src/platform/macos/capture_avfoundation.mm
        src/platform/macos/display_cocoa.mm
    )
    target_link_libraries(neurova PRIVATE
        "-framework AVFoundation"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework Cocoa"
    )
elseif(UNIX)
    target_sources(neurova PRIVATE
        src/platform/linux/capture_v4l2.cpp
        src/platform/linux/display_x11.cpp
    )
    if(NEUROVA_WITH_V4L2)
        target_link_libraries(neurova PRIVATE v4l2)
    endif()
endif()

# Python bindings
if(NEUROVA_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Tests
if(NEUROVA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(NEUROVA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Documentation
if(NEUROVA_BUILD_DOCS)
    add_subdirectory(doc)
endif()

# Install targets
include(GNUInstallDirs)

install(TARGETS neurova
    EXPORT NeurovaTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/neurova
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT NeurovaTargets
    FILE NeurovaTargets.cmake
    NAMESPACE Neurova::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Neurova
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeurovaConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/NeurovaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Neurova
)

write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/NeurovaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_BINARY_DIR}/NeurovaConfig.cmake"
    "${CMAKE_BINARY_DIR}/NeurovaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Neurova
)

# Print configuration summary
message(STATUS "")
message(STATUS "Neurova ${PROJECT_VERSION} Configuration Summary")
message(STATUS "===========================================")
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs:         ${NEUROVA_BUILD_SHARED_LIBS}")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "  Hardware Acceleration:")
message(STATUS "    CUDA:              ${NEUROVA_WITH_CUDA}")
message(STATUS "    OpenCL:            ${NEUROVA_WITH_OPENCL}")
message(STATUS "    Metal:             ${NEUROVA_WITH_METAL}")
message(STATUS "    Vulkan:            ${NEUROVA_WITH_VULKAN}")
message(STATUS "")
message(STATUS "  Math/Parallel:")
message(STATUS "    TBB:               ${NEUROVA_WITH_TBB}")
message(STATUS "    OpenMP:            ${NEUROVA_WITH_OPENMP}")
message(STATUS "    OpenBLAS:          ${NEUROVA_WITH_OPENBLAS}")
message(STATUS "    MKL:               ${NEUROVA_WITH_MKL}")
message(STATUS "    IPP:               ${NEUROVA_WITH_IPP}")
message(STATUS "")
message(STATUS "  Image Codecs:")
message(STATUS "    JPEG:              ${NEUROVA_WITH_JPEG}")
message(STATUS "    PNG:               ${NEUROVA_WITH_PNG}")
message(STATUS "    TIFF:              ${NEUROVA_WITH_TIFF}")
message(STATUS "    WebP:              ${NEUROVA_WITH_WEBP}")
message(STATUS "    OpenJPEG:          ${NEUROVA_WITH_OPENJPEG}")
message(STATUS "    OpenEXR:           ${NEUROVA_WITH_OPENEXR}")
message(STATUS "")
message(STATUS "  Video/Capture:")
message(STATUS "    FFmpeg:            ${NEUROVA_WITH_FFMPEG}")
message(STATUS "    V4L2:              ${NEUROVA_WITH_V4L2}")
message(STATUS "    Media Foundation:  ${NEUROVA_WITH_MSMF}")
message(STATUS "    AVFoundation:      ${NEUROVA_WITH_AVFOUNDATION}")
message(STATUS "")
message(STATUS "  GUI:")
message(STATUS "    Qt:                ${NEUROVA_WITH_QT}")
message(STATUS "    GTK:               ${NEUROVA_WITH_GTK}")
message(STATUS "    OpenGL:            ${NEUROVA_WITH_OPENGL}")
message(STATUS "")
