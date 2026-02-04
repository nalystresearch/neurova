# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# Neurova CMake Utilities
# Common utility functions for the build system

include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)
include(CheckSymbolExists)

# Detect platform
if(WIN32)
    set(NEUROVA_PLATFORM "Windows")
    set(NEUROVA_PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    set(NEUROVA_PLATFORM "macOS")
    set(NEUROVA_PLATFORM_MACOS TRUE)
elseif(UNIX)
    set(NEUROVA_PLATFORM "Linux")
    set(NEUROVA_PLATFORM_LINUX TRUE)
else()
    set(NEUROVA_PLATFORM "Unknown")
endif()

# Detect architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(NEUROVA_ARCH_64 TRUE)
    set(NEUROVA_ARCH "x64")
else()
    set(NEUROVA_ARCH_64 FALSE)
    set(NEUROVA_ARCH "x86")
endif()

# Check for ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(NEUROVA_ARCH_ARM TRUE)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(NEUROVA_ARCH "arm64")
    else()
        set(NEUROVA_ARCH "arm")
    endif()
endif()

# Function to add compiler flags if supported
function(neurova_add_cxx_flag flag)
    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" flag_var "HAVE_CXX_FLAG${flag}")
    check_cxx_compiler_flag("${flag}" ${flag_var})
    if(${flag_var})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
    endif()
endfunction()

# Function to detect SIMD support
function(neurova_detect_simd)
    set(NEUROVA_SIMD_FLAGS "" PARENT_SCOPE)
    
    # SSE
    check_cxx_compiler_flag("-msse" HAVE_SSE)
    check_cxx_compiler_flag("-msse2" HAVE_SSE2)
    check_cxx_compiler_flag("-msse3" HAVE_SSE3)
    check_cxx_compiler_flag("-mssse3" HAVE_SSSE3)
    check_cxx_compiler_flag("-msse4.1" HAVE_SSE41)
    check_cxx_compiler_flag("-msse4.2" HAVE_SSE42)
    
    # AVX
    check_cxx_compiler_flag("-mavx" HAVE_AVX)
    check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
    check_cxx_compiler_flag("-mavx512f" HAVE_AVX512)
    
    # NEON (ARM)
    check_cxx_compiler_flag("-mfpu=neon" HAVE_NEON)
    
    set(SIMD_FLAGS "")
    if(HAVE_SSE42)
        list(APPEND SIMD_FLAGS "-msse4.2")
    endif()
    if(HAVE_AVX2)
        list(APPEND SIMD_FLAGS "-mavx2")
    endif()
    if(HAVE_NEON)
        list(APPEND SIMD_FLAGS "-mfpu=neon")
    endif()
    
    set(NEUROVA_SIMD_FLAGS ${SIMD_FLAGS} PARENT_SCOPE)
    set(NEUROVA_HAVE_SSE ${HAVE_SSE} PARENT_SCOPE)
    set(NEUROVA_HAVE_SSE2 ${HAVE_SSE2} PARENT_SCOPE)
    set(NEUROVA_HAVE_AVX ${HAVE_AVX} PARENT_SCOPE)
    set(NEUROVA_HAVE_AVX2 ${HAVE_AVX2} PARENT_SCOPE)
    set(NEUROVA_HAVE_AVX512 ${HAVE_AVX512} PARENT_SCOPE)
    set(NEUROVA_HAVE_NEON ${HAVE_NEON} PARENT_SCOPE)
endfunction()

# Function to check for header availability
function(neurova_check_headers)
    check_include_file_cxx("immintrin.h" HAVE_IMMINTRIN_H)
    check_include_file_cxx("arm_neon.h" HAVE_ARM_NEON_H)
    check_include_file_cxx("pthread.h" HAVE_PTHREAD_H)
    check_include_file_cxx("sys/mman.h" HAVE_SYS_MMAN_H)
    check_include_file_cxx("dlfcn.h" HAVE_DLFCN_H)
    
    set(NEUROVA_HAVE_IMMINTRIN ${HAVE_IMMINTRIN_H} PARENT_SCOPE)
    set(NEUROVA_HAVE_ARM_NEON ${HAVE_ARM_NEON_H} PARENT_SCOPE)
    set(NEUROVA_HAVE_PTHREAD ${HAVE_PTHREAD_H} PARENT_SCOPE)
endfunction()

# Function to add a neurova module
function(neurova_add_module name)
    cmake_parse_arguments(MODULE
        ""
        ""
        "SOURCES;HEADERS;DEPENDENCIES"
        ${ARGN}
    )
    
    add_library(neurova_${name} OBJECT ${MODULE_SOURCES})
    
    target_include_directories(neurova_${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
    
    if(MODULE_DEPENDENCIES)
        target_link_libraries(neurova_${name} PRIVATE ${MODULE_DEPENDENCIES})
    endif()
    
    # Add to global list of modules
    set_property(GLOBAL APPEND PROPERTY NEUROVA_MODULES neurova_${name})
endfunction()

# Function to define GPU kernel
function(neurova_add_gpu_kernel name)
    cmake_parse_arguments(KERNEL
        ""
        "TYPE"
        "SOURCES"
        ${ARGN}
    )
    
    if(KERNEL_TYPE STREQUAL "CUDA")
        if(NEUROVA_WITH_CUDA)
            add_library(${name}_cuda OBJECT ${KERNEL_SOURCES})
            set_target_properties(${name}_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        endif()
    elseif(KERNEL_TYPE STREQUAL "OPENCL")
        # OpenCL kernels are compiled at runtime, just copy them
        foreach(src ${KERNEL_SOURCES})
            configure_file(${src} ${CMAKE_BINARY_DIR}/kernels/${src} COPYONLY)
        endforeach()
    elseif(KERNEL_TYPE STREQUAL "METAL")
        if(NEUROVA_WITH_METAL AND APPLE)
            # Metal shaders compiled with metallib
            foreach(src ${KERNEL_SOURCES})
                get_filename_component(name_we ${src} NAME_WE)
                add_custom_command(
                    OUTPUT ${CMAKE_BINARY_DIR}/kernels/${name_we}.air
                    COMMAND xcrun -sdk macosx metal -c ${src} -o ${CMAKE_BINARY_DIR}/kernels/${name_we}.air
                    DEPENDS ${src}
                )
                list(APPEND METAL_AIR_FILES ${CMAKE_BINARY_DIR}/kernels/${name_we}.air)
            endforeach()
        endif()
    endif()
endfunction()

# Macro to define compile definitions based on options
macro(neurova_define_options)
    if(NEUROVA_WITH_CUDA)
        add_compile_definitions(NEUROVA_HAVE_CUDA=1)
    endif()
    if(NEUROVA_WITH_OPENCL)
        add_compile_definitions(NEUROVA_HAVE_OPENCL=1)
    endif()
    if(NEUROVA_WITH_METAL)
        add_compile_definitions(NEUROVA_HAVE_METAL=1)
    endif()
    if(NEUROVA_WITH_VULKAN)
        add_compile_definitions(NEUROVA_HAVE_VULKAN=1)
    endif()
    if(NEUROVA_WITH_TBB)
        add_compile_definitions(NEUROVA_HAVE_TBB=1)
    endif()
    if(NEUROVA_WITH_OPENMP)
        add_compile_definitions(NEUROVA_HAVE_OPENMP=1)
    endif()
    if(NEUROVA_WITH_MKL)
        add_compile_definitions(NEUROVA_HAVE_MKL=1)
    endif()
    if(NEUROVA_WITH_IPP)
        add_compile_definitions(NEUROVA_HAVE_IPP=1)
    endif()
    if(NEUROVA_WITH_OPENBLAS)
        add_compile_definitions(NEUROVA_HAVE_OPENBLAS=1)
    endif()
endmacro()

# Print status helper
function(neurova_status msg)
    message(STATUS "[Neurova] ${msg}")
endfunction()

# Detect SIMD on configuration
neurova_detect_simd()
neurova_check_headers()
neurova_define_options()

message(STATUS "[Neurova] Platform: ${NEUROVA_PLATFORM} (${NEUROVA_ARCH})")
message(STATUS "[Neurova] SIMD support: SSE=${NEUROVA_HAVE_SSE} AVX=${NEUROVA_HAVE_AVX} AVX2=${NEUROVA_HAVE_AVX2} NEON=${NEUROVA_HAVE_NEON}")
