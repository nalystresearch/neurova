# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaOpenBLAS.cmake
# Locate OpenBLAS library

set(NEUROVA_OPENBLAS_FOUND FALSE)

find_package(BLAS QUIET)

if(BLAS_FOUND)
    # Check if it's OpenBLAS specifically
    find_path(OPENBLAS_INCLUDE_DIR cblas.h openblas/cblas.h
        PATHS /usr/include /usr/local/include
              /opt/OpenBLAS/include
              $ENV{OPENBLAS_ROOT}/include)
    
    if(OPENBLAS_INCLUDE_DIR)
        set(NEUROVA_OPENBLAS_FOUND TRUE)
        list(APPEND NEUROVA_EXTERNAL_LIBS ${BLAS_LIBRARIES})
        include_directories(${OPENBLAS_INCLUDE_DIR})
        add_compile_definitions(NEUROVA_HAVE_OPENBLAS=1)
        add_compile_definitions(NEUROVA_HAVE_BLAS=1)
        message(STATUS "[Neurova] OpenBLAS found")
    endif()
endif()

if(NOT NEUROVA_OPENBLAS_FOUND)
    # Manual search
    find_path(OPENBLAS_INCLUDE_DIR cblas.h
        PATHS /usr/include/openblas
              /usr/local/include/openblas
              /opt/OpenBLAS/include
              $ENV{OPENBLAS_ROOT}/include)
    
    find_library(OPENBLAS_LIBRARY openblas
        PATHS /usr/lib /usr/local/lib
              /opt/OpenBLAS/lib
              $ENV{OPENBLAS_ROOT}/lib)
    
    if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY)
        set(NEUROVA_OPENBLAS_FOUND TRUE)
        list(APPEND NEUROVA_EXTERNAL_LIBS ${OPENBLAS_LIBRARY})
        include_directories(${OPENBLAS_INCLUDE_DIR})
        add_compile_definitions(NEUROVA_HAVE_OPENBLAS=1)
        add_compile_definitions(NEUROVA_HAVE_BLAS=1)
        message(STATUS "[Neurova] OpenBLAS found (manual): ${OPENBLAS_LIBRARY}")
    else()
        message(WARNING "[Neurova] OpenBLAS not found")
    endif()
endif()

# Also find LAPACK if available
if(NEUROVA_OPENBLAS_FOUND)
    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
        list(APPEND NEUROVA_EXTERNAL_LIBS ${LAPACK_LIBRARIES})
        add_compile_definitions(NEUROVA_HAVE_LAPACK=1)
        message(STATUS "[Neurova] LAPACK found")
    else()
        # OpenBLAS often includes LAPACK
        find_library(LAPACK_LIBRARY lapack
            PATHS /usr/lib /usr/local/lib)
        if(LAPACK_LIBRARY)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${LAPACK_LIBRARY})
            add_compile_definitions(NEUROVA_HAVE_LAPACK=1)
            message(STATUS "[Neurova] LAPACK found (manual)")
        endif()
    endif()
endif()
