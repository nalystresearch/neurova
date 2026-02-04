# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaCUDA.cmake
# Locate CUDA toolkit and configure for Neurova

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    
    # Set CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Default to common architectures
        # 60 = Pascal, 70 = Volta, 75 = Turing, 80 = Ampere, 86 = Ampere, 89 = Ada, 90 = Hopper
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
    endif()
    
    # CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
    
    # Check for cuDNN
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        PATH_SUFFIXES include)
    find_library(CUDNN_LIBRARY cudnn
        HINTS ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 lib)
    
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        set(NEUROVA_HAVE_CUDNN TRUE)
        add_compile_definitions(NEUROVA_HAVE_CUDNN=1)
        list(APPEND NEUROVA_EXTERNAL_LIBS ${CUDNN_LIBRARY})
        message(STATUS "[Neurova] cuDNN found: ${CUDNN_LIBRARY}")
    else()
        set(NEUROVA_HAVE_CUDNN FALSE)
        message(STATUS "[Neurova] cuDNN not found (optional)")
    endif()
    
    # Check for cuBLAS
    if(TARGET CUDA::cublas)
        set(NEUROVA_HAVE_CUBLAS TRUE)
        add_compile_definitions(NEUROVA_HAVE_CUBLAS=1)
        list(APPEND NEUROVA_EXTERNAL_LIBS CUDA::cublas)
        message(STATUS "[Neurova] cuBLAS found")
    endif()
    
    # Check for cuFFT
    if(TARGET CUDA::cufft)
        set(NEUROVA_HAVE_CUFFT TRUE)
        add_compile_definitions(NEUROVA_HAVE_CUFFT=1)
        list(APPEND NEUROVA_EXTERNAL_LIBS CUDA::cufft)
        message(STATUS "[Neurova] cuFFT found")
    endif()
    
    # Check for NPP (NVIDIA Performance Primitives)
    if(TARGET CUDA::nppc)
        set(NEUROVA_HAVE_NPP TRUE)
        add_compile_definitions(NEUROVA_HAVE_NPP=1)
        list(APPEND NEUROVA_EXTERNAL_LIBS 
            CUDA::nppc 
            CUDA::nppig 
            CUDA::nppif 
            CUDA::nppist
            CUDA::nppidei
        )
        message(STATUS "[Neurova] NPP found")
    endif()
    
    # Check for NVJPEG
    find_library(NVJPEG_LIBRARY nvjpeg
        HINTS ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 lib)
    if(NVJPEG_LIBRARY)
        set(NEUROVA_HAVE_NVJPEG TRUE)
        add_compile_definitions(NEUROVA_HAVE_NVJPEG=1)
        list(APPEND NEUROVA_EXTERNAL_LIBS ${NVJPEG_LIBRARY})
        message(STATUS "[Neurova] nvJPEG found")
    endif()
    
    # Check for nvcomp
    find_library(NVCOMP_LIBRARY nvcomp
        HINTS ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 lib)
    if(NVCOMP_LIBRARY)
        set(NEUROVA_HAVE_NVCOMP TRUE)
        add_compile_definitions(NEUROVA_HAVE_NVCOMP=1)
        list(APPEND NEUROVA_EXTERNAL_LIBS ${NVCOMP_LIBRARY})
        message(STATUS "[Neurova] nvcomp found")
    endif()
    
    # Link CUDA runtime
    list(APPEND NEUROVA_EXTERNAL_LIBS CUDA::cudart)
    
    set(NEUROVA_CUDA_FOUND TRUE)
    add_compile_definitions(NEUROVA_HAVE_CUDA=1)
    
    message(STATUS "[Neurova] CUDA found: ${CUDAToolkit_VERSION}")
    message(STATUS "[Neurova] CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(NEUROVA_CUDA_FOUND FALSE)
    message(WARNING "[Neurova] CUDA compiler not found")
endif()

# Function to add CUDA sources to a target
function(neurova_add_cuda_sources target)
    if(NEUROVA_CUDA_FOUND)
        target_sources(${target} PRIVATE ${ARGN})
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    endif()
endfunction()

# Function to compile CUDA kernel library
function(neurova_cuda_kernel_library name)
    cmake_parse_arguments(KERNEL "" "" "SOURCES" ${ARGN})
    
    if(NEUROVA_CUDA_FOUND)
        add_library(${name} STATIC ${KERNEL_SOURCES})
        set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
        )
        target_link_libraries(${name} PRIVATE CUDA::cudart)
    endif()
endfunction()
