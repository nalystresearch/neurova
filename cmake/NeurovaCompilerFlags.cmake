# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# Neurova Compiler Flags Configuration
# Platform and compiler specific flags

include(CheckCXXCompilerFlag)

# Initialize flags
set(NEUROVA_CXX_FLAGS "")
set(NEUROVA_CXX_FLAGS_DEBUG "")
set(NEUROVA_CXX_FLAGS_RELEASE "")

# Compiler detection
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(NEUROVA_COMPILER "GCC")
    set(NEUROVA_COMPILER_GCC TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(NEUROVA_COMPILER "Clang")
    set(NEUROVA_COMPILER_CLANG TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(NEUROVA_COMPILER "MSVC")
    set(NEUROVA_COMPILER_MSVC TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(NEUROVA_COMPILER "Intel")
    set(NEUROVA_COMPILER_INTEL TRUE)
endif()

# GCC/Clang flags
if(NEUROVA_COMPILER_GCC OR NEUROVA_COMPILER_CLANG)
    # Common flags
    list(APPEND NEUROVA_CXX_FLAGS
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -fPIC
    )
    
    # Debug flags
    list(APPEND NEUROVA_CXX_FLAGS_DEBUG
        -g
        -O0
        -fno-omit-frame-pointer
        -DDEBUG
    )
    
    # Release flags
    list(APPEND NEUROVA_CXX_FLAGS_RELEASE
        -O3
        -DNDEBUG
        -ffast-math
        -funroll-loops
    )
    
    # Enable LTO for release
    check_cxx_compiler_flag("-flto" HAVE_LTO)
    if(HAVE_LTO)
        list(APPEND NEUROVA_CXX_FLAGS_RELEASE -flto)
    endif()
    
    # SIMD flags
    if(NEUROVA_HAVE_AVX2)
        list(APPEND NEUROVA_CXX_FLAGS_RELEASE -mavx2 -mfma)
    elseif(NEUROVA_HAVE_AVX)
        list(APPEND NEUROVA_CXX_FLAGS_RELEASE -mavx)
    elseif(NEUROVA_HAVE_SSE42)
        list(APPEND NEUROVA_CXX_FLAGS_RELEASE -msse4.2)
    endif()
    
    # ARM NEON
    if(NEUROVA_HAVE_NEON)
        list(APPEND NEUROVA_CXX_FLAGS_RELEASE -mfpu=neon)
    endif()
endif()

# MSVC flags
if(NEUROVA_COMPILER_MSVC)
    # Common flags
    list(APPEND NEUROVA_CXX_FLAGS
        /W4
        /EHsc
        /MP
        /utf-8
        /permissive-
    )
    
    # Debug flags
    list(APPEND NEUROVA_CXX_FLAGS_DEBUG
        /Od
        /Zi
        /RTC1
        /DEBUG
    )
    
    # Release flags
    list(APPEND NEUROVA_CXX_FLAGS_RELEASE
        /O2
        /Ob2
        /Oi
        /Ot
        /GL
        /DNDEBUG
    )
    
    # SIMD flags
    if(NEUROVA_HAVE_AVX2)
        list(APPEND NEUROVA_CXX_FLAGS_RELEASE /arch:AVX2)
    elseif(NEUROVA_HAVE_AVX)
        list(APPEND NEUROVA_CXX_FLAGS_RELEASE /arch:AVX)
    endif()
    
    # Disable warnings
    add_compile_options(/wd4100 /wd4244 /wd4267)
endif()

# Intel compiler flags
if(NEUROVA_COMPILER_INTEL)
    list(APPEND NEUROVA_CXX_FLAGS
        -Wall
        -fPIC
    )
    
    list(APPEND NEUROVA_CXX_FLAGS_DEBUG
        -g
        -O0
    )
    
    list(APPEND NEUROVA_CXX_FLAGS_RELEASE
        -O3
        -xHost
        -ipo
    )
endif()

# OpenMP flags
if(NEUROVA_WITH_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        list(APPEND NEUROVA_CXX_FLAGS ${OpenMP_CXX_FLAGS})
        list(APPEND NEUROVA_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
        message(STATUS "[Neurova] OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "[Neurova] OpenMP requested but not found")
        set(NEUROVA_WITH_OPENMP OFF)
    endif()
endif()

# Sanitizer support (Debug builds)
option(NEUROVA_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(NEUROVA_ENABLE_TSAN "Enable Thread Sanitizer" OFF)
option(NEUROVA_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)

if(NEUROVA_ENABLE_ASAN)
    list(APPEND NEUROVA_CXX_FLAGS_DEBUG -fsanitize=address)
    list(APPEND NEUROVA_LINK_FLAGS_DEBUG -fsanitize=address)
endif()

if(NEUROVA_ENABLE_TSAN)
    list(APPEND NEUROVA_CXX_FLAGS_DEBUG -fsanitize=thread)
    list(APPEND NEUROVA_LINK_FLAGS_DEBUG -fsanitize=thread)
endif()

if(NEUROVA_ENABLE_UBSAN)
    list(APPEND NEUROVA_CXX_FLAGS_DEBUG -fsanitize=undefined)
    list(APPEND NEUROVA_LINK_FLAGS_DEBUG -fsanitize=undefined)
endif()

# Apply flags
string(REPLACE ";" " " NEUROVA_CXX_FLAGS_STR "${NEUROVA_CXX_FLAGS}")
string(REPLACE ";" " " NEUROVA_CXX_FLAGS_DEBUG_STR "${NEUROVA_CXX_FLAGS_DEBUG}")
string(REPLACE ";" " " NEUROVA_CXX_FLAGS_RELEASE_STR "${NEUROVA_CXX_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NEUROVA_CXX_FLAGS_STR}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${NEUROVA_CXX_FLAGS_DEBUG_STR}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${NEUROVA_CXX_FLAGS_RELEASE_STR}")

message(STATUS "[Neurova] Compiler: ${NEUROVA_COMPILER} (${CMAKE_CXX_COMPILER_VERSION})")
