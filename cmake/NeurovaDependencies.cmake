# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# Neurova Dependencies Configuration
# Find and configure external dependencies

set(NEUROVA_EXTERNAL_LIBS "")
set(NEUROVA_PRIVATE_LIBS "")

# ============================================================================
# Image Codecs
# ============================================================================

# libjpeg-turbo
if(NEUROVA_WITH_JPEG)
    find_package(JPEG)
    if(JPEG_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS JPEG::JPEG)
        add_compile_definitions(NEUROVA_HAVE_JPEG=1)
        message(STATUS "[Neurova] JPEG found: ${JPEG_VERSION}")
    else()
        message(WARNING "[Neurova] JPEG requested but not found")
        set(NEUROVA_WITH_JPEG OFF)
    endif()
endif()

# libpng
if(NEUROVA_WITH_PNG)
    find_package(PNG)
    if(PNG_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS PNG::PNG)
        add_compile_definitions(NEUROVA_HAVE_PNG=1)
        message(STATUS "[Neurova] PNG found: ${PNG_VERSION_STRING}")
    else()
        message(WARNING "[Neurova] PNG requested but not found")
        set(NEUROVA_WITH_PNG OFF)
    endif()
endif()

# libtiff
if(NEUROVA_WITH_TIFF)
    find_package(TIFF)
    if(TIFF_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS TIFF::TIFF)
        add_compile_definitions(NEUROVA_HAVE_TIFF=1)
        message(STATUS "[Neurova] TIFF found")
    else()
        message(WARNING "[Neurova] TIFF requested but not found")
        set(NEUROVA_WITH_TIFF OFF)
    endif()
endif()

# libwebp
if(NEUROVA_WITH_WEBP)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(WEBP QUIET libwebp)
        if(WEBP_FOUND)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${WEBP_LIBRARIES})
            include_directories(${WEBP_INCLUDE_DIRS})
            add_compile_definitions(NEUROVA_HAVE_WEBP=1)
            message(STATUS "[Neurova] WebP found: ${WEBP_VERSION}")
        else()
            message(WARNING "[Neurova] WebP requested but not found")
            set(NEUROVA_WITH_WEBP OFF)
        endif()
    else()
        # Manual search
        find_path(WEBP_INCLUDE_DIR webp/decode.h)
        find_library(WEBP_LIBRARY webp)
        if(WEBP_INCLUDE_DIR AND WEBP_LIBRARY)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${WEBP_LIBRARY})
            include_directories(${WEBP_INCLUDE_DIR})
            add_compile_definitions(NEUROVA_HAVE_WEBP=1)
            message(STATUS "[Neurova] WebP found")
        else()
            message(WARNING "[Neurova] WebP requested but not found")
            set(NEUROVA_WITH_WEBP OFF)
        endif()
    endif()
endif()

# OpenJPEG (JPEG2000)
if(NEUROVA_WITH_OPENJPEG)
    find_package(OpenJPEG QUIET)
    if(OpenJPEG_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS openjp2)
        add_compile_definitions(NEUROVA_HAVE_OPENJPEG=1)
        message(STATUS "[Neurova] OpenJPEG found: ${OPENJPEG_VERSION}")
    else()
        message(WARNING "[Neurova] OpenJPEG requested but not found")
        set(NEUROVA_WITH_OPENJPEG OFF)
    endif()
endif()

# OpenEXR
if(NEUROVA_WITH_OPENEXR)
    find_package(OpenEXR QUIET)
    if(OpenEXR_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS OpenEXR::OpenEXR)
        add_compile_definitions(NEUROVA_HAVE_OPENEXR=1)
        message(STATUS "[Neurova] OpenEXR found")
    else()
        # Try Imath separately (OpenEXR 3.x)
        find_package(Imath QUIET)
        if(Imath_FOUND)
            list(APPEND NEUROVA_EXTERNAL_LIBS Imath::Imath)
            add_compile_definitions(NEUROVA_HAVE_OPENEXR=1)
            message(STATUS "[Neurova] OpenEXR (Imath) found")
        else()
            message(WARNING "[Neurova] OpenEXR requested but not found")
            set(NEUROVA_WITH_OPENEXR OFF)
        endif()
    endif()
endif()

# libavif
if(NEUROVA_WITH_AVIF)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(AVIF QUIET libavif)
        if(AVIF_FOUND)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${AVIF_LIBRARIES})
            include_directories(${AVIF_INCLUDE_DIRS})
            add_compile_definitions(NEUROVA_HAVE_AVIF=1)
            message(STATUS "[Neurova] AVIF found: ${AVIF_VERSION}")
        else()
            message(WARNING "[Neurova] AVIF requested but not found")
            set(NEUROVA_WITH_AVIF OFF)
        endif()
    else()
        message(WARNING "[Neurova] AVIF requested but pkg-config not found")
        set(NEUROVA_WITH_AVIF OFF)
    endif()
endif()

# JasPer
if(NEUROVA_WITH_JASPER)
    find_package(Jasper QUIET)
    if(Jasper_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS ${JASPER_LIBRARIES})
        include_directories(${JASPER_INCLUDE_DIR})
        add_compile_definitions(NEUROVA_HAVE_JASPER=1)
        message(STATUS "[Neurova] JasPer found")
    else()
        message(WARNING "[Neurova] JasPer requested but not found")
        set(NEUROVA_WITH_JASPER OFF)
    endif()
endif()

# ============================================================================
# Video/FFmpeg
# ============================================================================

if(NEUROVA_WITH_FFMPEG)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(AVCODEC QUIET libavcodec)
        pkg_check_modules(AVFORMAT QUIET libavformat)
        pkg_check_modules(AVUTIL QUIET libavutil)
        pkg_check_modules(SWSCALE QUIET libswscale)
        pkg_check_modules(SWRESAMPLE QUIET libswresample)
        
        if(AVCODEC_FOUND AND AVFORMAT_FOUND AND AVUTIL_FOUND)
            list(APPEND NEUROVA_EXTERNAL_LIBS 
                ${AVCODEC_LIBRARIES}
                ${AVFORMAT_LIBRARIES}
                ${AVUTIL_LIBRARIES}
            )
            if(SWSCALE_FOUND)
                list(APPEND NEUROVA_EXTERNAL_LIBS ${SWSCALE_LIBRARIES})
            endif()
            if(SWRESAMPLE_FOUND)
                list(APPEND NEUROVA_EXTERNAL_LIBS ${SWRESAMPLE_LIBRARIES})
            endif()
            
            include_directories(
                ${AVCODEC_INCLUDE_DIRS}
                ${AVFORMAT_INCLUDE_DIRS}
                ${AVUTIL_INCLUDE_DIRS}
            )
            
            add_compile_definitions(NEUROVA_HAVE_FFMPEG=1)
            message(STATUS "[Neurova] FFmpeg found: avcodec=${AVCODEC_VERSION}")
        else()
            message(WARNING "[Neurova] FFmpeg requested but not found")
            set(NEUROVA_WITH_FFMPEG OFF)
        endif()
    else()
        message(WARNING "[Neurova] FFmpeg requested but pkg-config not found")
        set(NEUROVA_WITH_FFMPEG OFF)
    endif()
endif()

# ============================================================================
# GUI Toolkits
# ============================================================================

# Qt
if(NEUROVA_WITH_QT)
    find_package(Qt6 QUIET COMPONENTS Widgets OpenGL)
    if(NOT Qt6_FOUND)
        find_package(Qt5 QUIET COMPONENTS Widgets OpenGL)
    endif()
    
    if(Qt6_FOUND OR Qt5_FOUND)
        if(Qt6_FOUND)
            list(APPEND NEUROVA_EXTERNAL_LIBS Qt6::Widgets)
            if(TARGET Qt6::OpenGL)
                list(APPEND NEUROVA_EXTERNAL_LIBS Qt6::OpenGL)
            endif()
            message(STATUS "[Neurova] Qt6 found")
        else()
            list(APPEND NEUROVA_EXTERNAL_LIBS Qt5::Widgets)
            if(TARGET Qt5::OpenGL)
                list(APPEND NEUROVA_EXTERNAL_LIBS Qt5::OpenGL)
            endif()
            message(STATUS "[Neurova] Qt5 found")
        endif()
        add_compile_definitions(NEUROVA_HAVE_QT=1)
    else()
        message(WARNING "[Neurova] Qt requested but not found")
        set(NEUROVA_WITH_QT OFF)
    endif()
endif()

# GTK
if(NEUROVA_WITH_GTK)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(GTK3 QUIET gtk+-3.0)
        if(GTK3_FOUND)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${GTK3_LIBRARIES})
            include_directories(${GTK3_INCLUDE_DIRS})
            add_compile_definitions(NEUROVA_HAVE_GTK=1)
            message(STATUS "[Neurova] GTK3 found: ${GTK3_VERSION}")
        else()
            message(WARNING "[Neurova] GTK requested but not found")
            set(NEUROVA_WITH_GTK OFF)
        endif()
    else()
        message(WARNING "[Neurova] GTK requested but pkg-config not found")
        set(NEUROVA_WITH_GTK OFF)
    endif()
endif()

# OpenGL
if(NEUROVA_WITH_OPENGL)
    find_package(OpenGL)
    if(OpenGL_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS OpenGL::GL)
        if(TARGET OpenGL::GLU)
            list(APPEND NEUROVA_EXTERNAL_LIBS OpenGL::GLU)
        endif()
        add_compile_definitions(NEUROVA_HAVE_OPENGL=1)
        message(STATUS "[Neurova] OpenGL found")
    else()
        message(WARNING "[Neurova] OpenGL requested but not found")
        set(NEUROVA_WITH_OPENGL OFF)
    endif()
endif()

# ============================================================================
# Serialization
# ============================================================================

if(NEUROVA_WITH_PROTOBUF)
    find_package(Protobuf QUIET)
    if(Protobuf_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS protobuf::libprotobuf)
        add_compile_definitions(NEUROVA_HAVE_PROTOBUF=1)
        message(STATUS "[Neurova] Protobuf found: ${Protobuf_VERSION}")
    else()
        message(WARNING "[Neurova] Protobuf requested but not found")
        set(NEUROVA_WITH_PROTOBUF OFF)
    endif()
endif()

if(NEUROVA_WITH_FLATBUFFERS)
    find_package(flatbuffers QUIET)
    if(flatbuffers_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS flatbuffers::flatbuffers)
        add_compile_definitions(NEUROVA_HAVE_FLATBUFFERS=1)
        message(STATUS "[Neurova] FlatBuffers found")
    else()
        message(WARNING "[Neurova] FlatBuffers requested but not found")
        set(NEUROVA_WITH_FLATBUFFERS OFF)
    endif()
endif()

# ============================================================================
# DNN Runtime
# ============================================================================

if(NEUROVA_WITH_ONNX)
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS onnxruntime)
        add_compile_definitions(NEUROVA_HAVE_ONNX=1)
        message(STATUS "[Neurova] ONNX Runtime found")
    else()
        # Manual search
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATH_SUFFIXES onnxruntime/core/session)
        find_library(ONNXRUNTIME_LIBRARY onnxruntime)
        if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${ONNXRUNTIME_LIBRARY})
            include_directories(${ONNXRUNTIME_INCLUDE_DIR})
            add_compile_definitions(NEUROVA_HAVE_ONNX=1)
            message(STATUS "[Neurova] ONNX Runtime found (manual)")
        else()
            message(WARNING "[Neurova] ONNX Runtime requested but not found")
            set(NEUROVA_WITH_ONNX OFF)
        endif()
    endif()
endif()

# ============================================================================
# Platform-specific capture backends
# ============================================================================

# V4L2 (Linux)
if(NEUROVA_WITH_V4L2 AND UNIX AND NOT APPLE)
    find_path(V4L2_INCLUDE_DIR linux/videodev2.h)
    find_library(V4L2_LIBRARY v4l2)
    if(V4L2_INCLUDE_DIR)
        add_compile_definitions(NEUROVA_HAVE_V4L2=1)
        if(V4L2_LIBRARY)
            list(APPEND NEUROVA_PRIVATE_LIBS ${V4L2_LIBRARY})
        endif()
        message(STATUS "[Neurova] V4L2 found")
    else()
        message(WARNING "[Neurova] V4L2 requested but not found")
        set(NEUROVA_WITH_V4L2 OFF)
    endif()
endif()

# Media Foundation (Windows)
if(NEUROVA_WITH_MSMF AND WIN32)
    add_compile_definitions(NEUROVA_HAVE_MSMF=1)
    list(APPEND NEUROVA_PRIVATE_LIBS mfplat mfreadwrite mfuuid)
    message(STATUS "[Neurova] Media Foundation enabled")
endif()

# AVFoundation (macOS)
if(NEUROVA_WITH_AVFOUNDATION AND APPLE)
    add_compile_definitions(NEUROVA_HAVE_AVFOUNDATION=1)
    message(STATUS "[Neurova] AVFoundation enabled")
endif()

# pybind11 for Python bindings
if(NEUROVA_BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "[Neurova] pybind11 found: ${pybind11_VERSION}")
    else()
        # Try fetching pybind11
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
        message(STATUS "[Neurova] pybind11 fetched")
    endif()
endif()
