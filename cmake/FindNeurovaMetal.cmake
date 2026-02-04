# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaMetal.cmake
# Locate Metal framework and configure for Neurova (macOS/iOS)

if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders)
    
    if(METAL_FRAMEWORK)
        set(NEUROVA_METAL_FOUND TRUE)
        
        list(APPEND NEUROVA_EXTERNAL_LIBS 
            ${METAL_FRAMEWORK}
            ${METALKIT_FRAMEWORK}
        )
        
        if(METALPERFORMANCE_FRAMEWORK)
            list(APPEND NEUROVA_EXTERNAL_LIBS ${METALPERFORMANCE_FRAMEWORK})
            add_compile_definitions(NEUROVA_HAVE_MPS=1)
            message(STATUS "[Neurova] Metal Performance Shaders found")
        endif()
        
        add_compile_definitions(NEUROVA_HAVE_METAL=1)
        
        # Find Metal compiler
        find_program(METAL_COMPILER xcrun)
        
        message(STATUS "[Neurova] Metal framework found")
        
        # Create shader output directory
        set(NEUROVA_METAL_SHADER_DIR "${CMAKE_BINARY_DIR}/shaders/metal")
        file(MAKE_DIRECTORY ${NEUROVA_METAL_SHADER_DIR})
        
    else()
        set(NEUROVA_METAL_FOUND FALSE)
        message(WARNING "[Neurova] Metal framework not found")
    endif()
else()
    set(NEUROVA_METAL_FOUND FALSE)
    message(STATUS "[Neurova] Metal is only available on Apple platforms")
endif()

# Function to compile Metal shaders
function(neurova_compile_metal_shaders target)
    cmake_parse_arguments(SHADER "" "OUTPUT_DIR" "SOURCES" ${ARGN})
    
    if(NEUROVA_METAL_FOUND AND METAL_COMPILER)
        if(NOT SHADER_OUTPUT_DIR)
            set(SHADER_OUTPUT_DIR ${NEUROVA_METAL_SHADER_DIR})
        endif()
        
        set(AIR_FILES "")
        foreach(shader_src ${SHADER_SOURCES})
            get_filename_component(shader_name ${shader_src} NAME_WE)
            set(air_file ${SHADER_OUTPUT_DIR}/${shader_name}.air)
            
            add_custom_command(
                OUTPUT ${air_file}
                COMMAND ${METAL_COMPILER} -sdk macosx metal
                    -c ${shader_src}
                    -o ${air_file}
                DEPENDS ${shader_src}
                COMMENT "Compiling Metal shader ${shader_name}"
            )
            list(APPEND AIR_FILES ${air_file})
        endforeach()
        
        # Link into metallib
        set(METALLIB_FILE ${SHADER_OUTPUT_DIR}/${target}.metallib)
        add_custom_command(
            OUTPUT ${METALLIB_FILE}
            COMMAND ${METAL_COMPILER} -sdk macosx metallib
                ${AIR_FILES}
                -o ${METALLIB_FILE}
            DEPENDS ${AIR_FILES}
            COMMENT "Linking Metal library ${target}"
        )
        
        add_custom_target(${target}_metallib ALL
            DEPENDS ${METALLIB_FILE}
        )
        
        # Install metallib
        install(FILES ${METALLIB_FILE}
            DESTINATION share/neurova/shaders)
    endif()
endfunction()

# Function to add Metal source files
function(neurova_add_metal_sources target)
    if(NEUROVA_METAL_FOUND)
        foreach(src ${ARGN})
            set_source_files_properties(${src} PROPERTIES
                COMPILE_FLAGS "-x objective-c++"
            )
        endforeach()
        target_sources(${target} PRIVATE ${ARGN})
    endif()
endfunction()
