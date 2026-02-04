# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaVulkan.cmake
# Locate Vulkan SDK and configure compute shaders for Neurova

find_package(Vulkan QUIET)

if(Vulkan_FOUND)
    set(NEUROVA_VULKAN_FOUND TRUE)
    
    list(APPEND NEUROVA_EXTERNAL_LIBS Vulkan::Vulkan)
    add_compile_definitions(NEUROVA_HAVE_VULKAN=1)
    
    # Check for glslc shader compiler
    find_program(GLSLC glslc
        HINTS ${Vulkan_INCLUDE_DIRS}/../Bin
              $ENV{VULKAN_SDK}/bin
              $ENV{VK_SDK_PATH}/bin)
    
    if(GLSLC)
        set(NEUROVA_HAVE_GLSLC TRUE)
        message(STATUS "[Neurova] glslc shader compiler found: ${GLSLC}")
    else()
        set(NEUROVA_HAVE_GLSLC FALSE)
        message(STATUS "[Neurova] glslc not found (shader compilation unavailable)")
    endif()
    
    # Check for validation layers (debug)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_definitions(NEUROVA_VULKAN_VALIDATION=1)
    endif()
    
    message(STATUS "[Neurova] Vulkan found: ${Vulkan_VERSION}")
    message(STATUS "[Neurova] Vulkan include: ${Vulkan_INCLUDE_DIRS}")
    
    # Create shader output directory
    set(NEUROVA_VULKAN_SHADER_DIR "${CMAKE_BINARY_DIR}/shaders/vulkan")
    file(MAKE_DIRECTORY ${NEUROVA_VULKAN_SHADER_DIR})
    
else()
    set(NEUROVA_VULKAN_FOUND FALSE)
    message(WARNING "[Neurova] Vulkan SDK not found")
endif()

# Function to compile GLSL compute shaders to SPIR-V
function(neurova_compile_vulkan_shaders target)
    cmake_parse_arguments(SHADER "" "OUTPUT_DIR" "SOURCES" ${ARGN})
    
    if(NEUROVA_VULKAN_FOUND AND NEUROVA_HAVE_GLSLC)
        if(NOT SHADER_OUTPUT_DIR)
            set(SHADER_OUTPUT_DIR ${NEUROVA_VULKAN_SHADER_DIR})
        endif()
        
        set(SPV_FILES "")
        foreach(shader_src ${SHADER_SOURCES})
            get_filename_component(shader_name ${shader_src} NAME)
            set(spv_file ${SHADER_OUTPUT_DIR}/${shader_name}.spv)
            
            add_custom_command(
                OUTPUT ${spv_file}
                COMMAND ${GLSLC}
                    -fshader-stage=compute
                    -O
                    ${shader_src}
                    -o ${spv_file}
                DEPENDS ${shader_src}
                COMMENT "Compiling Vulkan shader ${shader_name}"
            )
            list(APPEND SPV_FILES ${spv_file})
        endforeach()
        
        add_custom_target(${target}_shaders ALL
            DEPENDS ${SPV_FILES}
        )
        
        # Install SPIR-V shaders
        install(FILES ${SPV_FILES}
            DESTINATION share/neurova/shaders/vulkan)
    endif()
endfunction()

# Function to compile HLSL to SPIR-V (using DXC)
function(neurova_compile_hlsl_shaders target)
    cmake_parse_arguments(SHADER "" "OUTPUT_DIR" "SOURCES" ${ARGN})
    
    # Find DXC compiler
    find_program(DXC dxc
        HINTS ${Vulkan_INCLUDE_DIRS}/../Bin
              $ENV{VULKAN_SDK}/bin)
    
    if(DXC)
        if(NOT SHADER_OUTPUT_DIR)
            set(SHADER_OUTPUT_DIR ${NEUROVA_VULKAN_SHADER_DIR})
        endif()
        
        set(SPV_FILES "")
        foreach(shader_src ${SHADER_SOURCES})
            get_filename_component(shader_name ${shader_src} NAME_WE)
            set(spv_file ${SHADER_OUTPUT_DIR}/${shader_name}.spv)
            
            add_custom_command(
                OUTPUT ${spv_file}
                COMMAND ${DXC}
                    -T cs_6_0
                    -spirv
                    -O3
                    ${shader_src}
                    -Fo ${spv_file}
                DEPENDS ${shader_src}
                COMMENT "Compiling HLSL shader ${shader_name} to SPIR-V"
            )
            list(APPEND SPV_FILES ${spv_file})
        endforeach()
        
        add_custom_target(${target}_hlsl_shaders ALL
            DEPENDS ${SPV_FILES}
        )
    else()
        message(STATUS "[Neurova] DXC not found, HLSL compilation unavailable")
    endif()
endfunction()
