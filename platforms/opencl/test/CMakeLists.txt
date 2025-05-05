find_package(OpenCL REQUIRED)

# opencl common
file(GLOB_RECURSE OPENCL_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(opencl_common OBJECT ${OPENCL_COMMON_SRCS})
target_include_directories(opencl_common PRIVATE
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)
target_compile_options(opencl_common PRIVATE
    -Wall
    -Wextra
    -DCL_TARGET_OPENCL_VERSION=300
)
target_link_libraries(opencl_common
    PRIVATE utils
    PUBLIC  OpenCL::OpenCL
)

# opencl main
file(GLOB_RECURSE OPENCL_MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)
foreach(OPENCL_MAIN_SRC ${OPENCL_MAIN_SRCS})
    get_filename_component(OPENCL_TARGET ${OPENCL_MAIN_SRC} NAME_WE)
    add_executable(${OPENCL_TARGET}
        ${OPENCL_MAIN_SRC}
        $<TARGET_OBJECTS:opencl_common>
    )
    target_include_directories(${OPENCL_TARGET} PRIVATE
        ${XSCHED_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
    )
    target_compile_options(${OPENCL_TARGET} PRIVATE
        -Wall
        -Wextra
        -DCL_TARGET_OPENCL_VERSION=300
    )
    target_link_libraries(${OPENCL_TARGET} PRIVATE
        utils
        preempt
        halopencl
        OpenCL::OpenCL
    )
    install(TARGETS ${OPENCL_TARGET} RUNTIME DESTINATION bin/opencl)
    set_target_properties(${OPENCL_TARGET} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endforeach()
