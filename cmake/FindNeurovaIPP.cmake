# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaIPP.cmake
# Locate Intel Integrated Performance Primitives

set(NEUROVA_IPP_FOUND FALSE)

# Try official CMake config
find_package(IPP QUIET CONFIG)

if(IPP_FOUND)
    set(NEUROVA_IPP_FOUND TRUE)
    list(APPEND NEUROVA_EXTERNAL_LIBS IPP::ipp)
    add_compile_definitions(NEUROVA_HAVE_IPP=1)
    message(STATUS "[Neurova] IPP found via CMake config")
else()
    # Manual search
    set(IPP_ROOT_HINTS
        $ENV{IPPROOT}
        $ENV{IPP_ROOT}
        /opt/intel/oneapi/ipp/latest
        /opt/intel/ipp
        "C:/Program Files (x86)/Intel/oneAPI/ipp/latest"
    )
    
    find_path(IPP_INCLUDE_DIR ipp.h
        HINTS ${IPP_ROOT_HINTS}
        PATH_SUFFIXES include)
    
    # Find IPP libraries
    # Core libraries
    find_library(IPPCORE_LIB ippcore
        HINTS ${IPP_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    find_library(IPPI_LIB ippi
        HINTS ${IPP_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    find_library(IPPS_LIB ipps
        HINTS ${IPP_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    find_library(IPPCC_LIB ippcc
        HINTS ${IPP_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    find_library(IPPCV_LIB ippcv
        HINTS ${IPP_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    find_library(IPPVM_LIB ippvm
        HINTS ${IPP_ROOT_HINTS}
        PATH_SUFFIXES lib lib/intel64)
    
    if(IPP_INCLUDE_DIR AND IPPCORE_LIB AND IPPI_LIB)
        set(NEUROVA_IPP_FOUND TRUE)
        
        list(APPEND NEUROVA_EXTERNAL_LIBS
            ${IPPI_LIB}
            ${IPPS_LIB}
            ${IPPCC_LIB}
            ${IPPCV_LIB}
            ${IPPVM_LIB}
            ${IPPCORE_LIB}
        )
        
        include_directories(${IPP_INCLUDE_DIR})
        add_compile_definitions(NEUROVA_HAVE_IPP=1)
        
        # Check for IPP Integrated Primitives for Deep Learning
        find_library(IPPIDL_LIB ippidl
            HINTS ${IPP_ROOT_HINTS}
            PATH_SUFFIXES lib lib/intel64)
        if(IPPIDL_LIB)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${IPPIDL_LIB})
            add_compile_definitions(NEUROVA_HAVE_IPP_DL=1)
            message(STATUS "[Neurova] IPP Deep Learning found")
        endif()
        
        message(STATUS "[Neurova] IPP found (manual)")
        message(STATUS "[Neurova] IPP include: ${IPP_INCLUDE_DIR}")
    else()
        message(WARNING "[Neurova] IPP not found")
    endif()
endif()
