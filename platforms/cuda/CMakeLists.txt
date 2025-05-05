# cuda hal
if(NOT CUDA_GEN_CODE)
    execute_process(
        COMMAND which nvidia-smi
        RESULT_VARIABLE NVIDIA_SMI_NOT_FOUND
        OUTPUT_QUIET
        ERROR_QUIET)
    if(NVIDIA_SMI_NOT_FOUND)
        message(WARNING "nvidia-smi not found")
        message(FATAL_ERROR "Error: gpu generation code not set! Pass by -DCUDA_GEN_CODE=xx")
    endif()

    execute_process(
        COMMAND sh -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1"
        RESULT_VARIABLE NVIDIA_SMI_STATUS
        OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NVIDIA_SMI_STATUS EQUAL 0)
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+)" "\\1\\2" CUDA_GEN_CODE ${CUDA_COMPUTE_CAPABILITY})
        message(STATUS "Detected GPU generation code: ${CUDA_GEN_CODE}")
    else()
        message(WARNING "nvidia-smi failed with ${NVIDIA_SMI_STATUS}")
        message(FATAL_ERROR "Error: gpu generation code not set! Pass by -DCUDA_GEN_CODE=xx")
    endif()
endif()

set(ARCH_RELATED_INSTRUMENT_SRC
    ${CMAKE_CURRENT_LIST_DIR}/hal/src/arch/sm${CUDA_GEN_CODE}.cpp
)

if(NOT EXISTS ${ARCH_RELATED_INSTRUMENT_SRC})
    message(WARNING "Warning: specific architecture implementation for sm${CUDA_GEN_CODE} not found, "
                    "using default implementation @ Level-1")
    set(ARCH_RELATED_INSTRUMENT_SRC ${CMAKE_CURRENT_LIST_DIR}/hal/src/arch/default.cpp)
endif()

file(GLOB_RECURSE HAL_CUDA_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/hal/src/level*/*.c
    ${CMAKE_CURRENT_LIST_DIR}/hal/src/level*/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hal/src/common/*.c
    ${CMAKE_CURRENT_LIST_DIR}/hal/src/common/*.cpp
)

add_library(halcuda SHARED
    ${HAL_CUDA_SRCS}
    ${ARCH_RELATED_INSTRUMENT_SRC}
)
add_library(XSched::halcuda ALIAS halcuda)

target_include_directories(halcuda PUBLIC
    $<BUILD_INTERFACE:${XSCHED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hal/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(halcuda PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(halcuda
    PRIVATE cuxtra
    PRIVATE utils
    PRIVATE protocol
    PUBLIC  preempt
)

set_target_properties(halcuda PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS halcuda EXPORT XSchedTargets LIBRARY DESTINATION lib)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/hal/include/xsched/cuda/hal.h DESTINATION include/xsched/cuda)

# cuda shim
file(GLOB_RECURSE SHIM_CUDA_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/shim/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/shim/src/*.cpp
)

add_library(shimcuda SHARED
    ${SHIM_CUDA_SRCS}
)
add_library(XSched::shimcuda ALIAS shimcuda)

target_include_directories(shimcuda PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/shim/include
)

target_compile_options(shimcuda PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
    -fvisibility=hidden
)

target_link_libraries(shimcuda
    PRIVATE utils
    PRIVATE protocol
    PUBLIC  halcuda
    PUBLIC  preempt
)

target_link_options(shimcuda PRIVATE
    -Wl,--exclude-libs,ALL
)

set_target_properties(shimcuda PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS shimcuda LIBRARY DESTINATION lib)

if(SHIM_SOFTLINK)

    set(SHIM_CUDA_SOFTLINKS
        libcuda.so.1
        libcuda.so
    )

    foreach(SOFTLINK ${SHIM_CUDA_SOFTLINKS})
        install(CODE "
            execute_process(
                COMMAND ln -s libshimcuda.so ${SOFTLINK}
                WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
            )
            message(\"-- Installing: softlink created: ${SOFTLINK}\")
        ")
    endforeach()

endif()

# test
if (BUILD_TEST)
    add_subdirectory(test)
endif()
