# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaTBB.cmake
# Locate Intel Threading Building Blocks

find_package(TBB QUIET)

if(TBB_FOUND)
    set(NEUROVA_TBB_FOUND TRUE)
    
    list(APPEND NEUROVA_EXTERNAL_LIBS TBB::tbb)
    add_compile_definitions(NEUROVA_HAVE_TBB=1)
    
    message(STATUS "[Neurova] TBB found: ${TBB_VERSION}")
else()
    # Try pkg-config
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(TBB QUIET tbb)
        if(TBB_FOUND)
            set(NEUROVA_TBB_FOUND TRUE)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${TBB_LIBRARIES})
            include_directories(${TBB_INCLUDE_DIRS})
            add_compile_definitions(NEUROVA_HAVE_TBB=1)
            message(STATUS "[Neurova] TBB found via pkg-config: ${TBB_VERSION}")
        endif()
    endif()
    
    if(NOT NEUROVA_TBB_FOUND)
        # Manual search
        find_path(TBB_INCLUDE_DIR tbb/tbb.h
            PATHS /usr/include /usr/local/include
                  /opt/intel/oneapi/tbb/latest/include
                  $ENV{TBBROOT}/include)
        find_library(TBB_LIBRARY tbb
            PATHS /usr/lib /usr/local/lib
                  /opt/intel/oneapi/tbb/latest/lib
                  $ENV{TBBROOT}/lib)
        
        if(TBB_INCLUDE_DIR AND TBB_LIBRARY)
            set(NEUROVA_TBB_FOUND TRUE)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${TBB_LIBRARY})
            include_directories(${TBB_INCLUDE_DIR})
            add_compile_definitions(NEUROVA_HAVE_TBB=1)
            message(STATUS "[Neurova] TBB found (manual)")
        else()
            set(NEUROVA_TBB_FOUND FALSE)
            message(WARNING "[Neurova] TBB not found")
        endif()
    endif()
endif()
