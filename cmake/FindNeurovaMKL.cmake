# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaMKL.cmake
# Locate Intel Math Kernel Library

set(NEUROVA_MKL_FOUND FALSE)

# Try official CMake config
find_package(MKL QUIET CONFIG)

if(MKL_FOUND)
    set(NEUROVA_MKL_FOUND TRUE)
    list(APPEND NEUROVA_EXTERNAL_LIBS MKL::MKL)
    add_compile_definitions(NEUROVA_HAVE_MKL=1)
    message(STATUS "[Neurova] MKL found via CMake config")
else()
    # Manual search
    set(MKL_ROOT_HINTS
        $ENV{MKLROOT}
        $ENV{MKL_ROOT}
        /opt/intel/oneapi/mkl/latest
        /opt/intel/mkl
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
    )
    
    find_path(MKL_INCLUDE_DIR mkl.h
        HINTS ${MKL_ROOT_HINTS}
        PATH_SUFFIXES include)
    
    # Find libraries based on linking model
    # Default to LP64 (32-bit integers) and sequential threading
    find_library(MKL_CORE_LIB mkl_core
        HINTS ${MKL_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    find_library(MKL_INTEL_LIB mkl_intel_lp64
        HINTS ${MKL_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    # Threading library (sequential, TBB, or OpenMP)
    if(NEUROVA_WITH_TBB AND NEUROVA_TBB_FOUND)
        find_library(MKL_THREAD_LIB mkl_tbb_thread
            HINTS ${MKL_ROOT_HINTS}
            PATH_SUFFIXES lib lib/intel64)
    elseif(NEUROVA_WITH_OPENMP)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            find_library(MKL_THREAD_LIB mkl_gnu_thread
                HINTS ${MKL_ROOT_HINTS}
                PATH_SUFFIXES lib lib/intel64)
        else()
            find_library(MKL_THREAD_LIB mkl_intel_thread
                HINTS ${MKL_ROOT_HINTS}
                PATH_SUFFIXES lib lib/intel64)
        endif()
    else()
        find_library(MKL_THREAD_LIB mkl_sequential
            HINTS ${MKL_ROOT_HINTS}
            PATH_SUFFIXES lib lib/intel64)
    endif()
    
    if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_INTEL_LIB AND MKL_THREAD_LIB)
        set(NEUROVA_MKL_FOUND TRUE)
        
        list(APPEND NEUROVA_EXTERNAL_LIBS
            ${MKL_INTEL_LIB}
            ${MKL_THREAD_LIB}
            ${MKL_CORE_LIB}
        )
        
        # Link math library on Unix
        if(UNIX)
            list(APPEND NEUROVA_EXTERNAL_LIBS m pthread dl)
        endif()
        
        include_directories(${MKL_INCLUDE_DIR})
        add_compile_definitions(NEUROVA_HAVE_MKL=1)
        
        message(STATUS "[Neurova] MKL found (manual)")
        message(STATUS "[Neurova] MKL include: ${MKL_INCLUDE_DIR}")
    else()
        message(WARNING "[Neurova] MKL not found")
    endif()
endif()

# MKL provides BLAS and LAPACK
if(NEUROVA_MKL_FOUND)
    set(NEUROVA_HAVE_BLAS TRUE)
    set(NEUROVA_HAVE_LAPACK TRUE)
    add_compile_definitions(NEUROVA_HAVE_BLAS=1)
    add_compile_definitions(NEUROVA_HAVE_LAPACK=1)
endif()
