# copyright (c) 2025 @analytics withharry
# all rights reserved.
# licensed under the mit license.

# FindNeurovaOpenCL.cmake
# Locate OpenCL and configure for Neurova

find_package(OpenCL QUIET)

if(OpenCL_FOUND)
    set(NEUROVA_OPENCL_FOUND TRUE)
    
    list(APPEND NEUROVA_EXTERNAL_LIBS OpenCL::OpenCL)
    add_compile_definitions(NEUROVA_HAVE_OPENCL=1)
    
    # Check OpenCL version
    if(OpenCL_VERSION_MAJOR)
        add_compile_definitions(NEUROVA_OPENCL_VERSION_MAJOR=${OpenCL_VERSION_MAJOR})
        add_compile_definitions(NEUROVA_OPENCL_VERSION_MINOR=${OpenCL_VERSION_MINOR})
    endif()
    
    message(STATUS "[Neurova] OpenCL found: ${OpenCL_VERSION_STRING}")
    message(STATUS "[Neurova] OpenCL include: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "[Neurova] OpenCL library: ${OpenCL_LIBRARIES}")
    
    # Create kernel output directory
    set(NEUROVA_OPENCL_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels/opencl")
    file(MAKE_DIRECTORY ${NEUROVA_OPENCL_KERNEL_DIR})
    
else()
    set(NEUROVA_OPENCL_FOUND FALSE)
    message(WARNING "[Neurova] OpenCL not found")
endif()

# Function to install OpenCL kernels
function(neurova_install_opencl_kernels)
    cmake_parse_arguments(KERNEL "" "DESTINATION" "SOURCES" ${ARGN})
    
    if(NEUROVA_OPENCL_FOUND)
        foreach(kernel_src ${KERNEL_SOURCES})
            get_filename_component(kernel_name ${kernel_src} NAME)
            configure_file(
                ${kernel_src}
                ${NEUROVA_OPENCL_KERNEL_DIR}/${kernel_name}
                COPYONLY
            )
        endforeach()
        
        # Install kernels
        if(KERNEL_DESTINATION)
            install(FILES ${KERNEL_SOURCES}
                DESTINATION ${KERNEL_DESTINATION})
        else()
            install(FILES ${KERNEL_SOURCES}
                DESTINATION share/neurova/kernels/opencl)
        endif()
    endif()
endfunction()

# Function to compile OpenCL C code to SPIR-V (if available)
function(neurova_compile_opencl_spirv)
    cmake_parse_arguments(SPIRV "" "OUTPUT" "SOURCES" ${ARGN})
    
    if(NEUROVA_OPENCL_FOUND)
        # Look for clang with OpenCL support
        find_program(CLANG_OPENCL clang)
        find_program(LLVM_SPIRV llvm-spirv)
        
        if(CLANG_OPENCL AND LLVM_SPIRV)
            foreach(src ${SPIRV_SOURCES})
                get_filename_component(name_we ${src} NAME_WE)
                set(bc_file ${CMAKE_BINARY_DIR}/kernels/${name_we}.bc)
                set(spv_file ${CMAKE_BINARY_DIR}/kernels/${name_we}.spv)
                
                add_custom_command(
                    OUTPUT ${spv_file}
                    COMMAND ${CLANG_OPENCL} -c -target spir64 -O2 -emit-llvm
                        -o ${bc_file} ${src}
                    COMMAND ${LLVM_SPIRV} ${bc_file} -o ${spv_file}
                    DEPENDS ${src}
                    COMMENT "Compiling OpenCL kernel ${name_we} to SPIR-V"
                )
                list(APPEND SPIRV_OUTPUTS ${spv_file})
            endforeach()
            
            set(${SPIRV_OUTPUT} ${SPIRV_OUTPUTS} PARENT_SCOPE)
        else()
            message(STATUS "[Neurova] SPIR-V compilation not available (clang/llvm-spirv not found)")
        endif()
    endif()
endfunction()
