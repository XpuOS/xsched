enable_language(CUDA)

# cudart common
file(GLOB_RECURSE CUDART_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(cudart_common OBJECT ${CUDART_COMMON_SRCS})
target_include_directories(cudart_common PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)
target_link_libraries(cudart_common PRIVATE
    ${CUDA_LIBRARIES}
    utils
)

# cudart main
file(GLOB_RECURSE CUDART_MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)
foreach(CUDART_MAIN_SRC ${CUDART_MAIN_SRCS})
    get_filename_component(CUDART_TARGET ${CUDART_MAIN_SRC} NAME_WE)
    add_executable(${CUDART_TARGET}
        ${CUDART_MAIN_SRC}
        $<TARGET_OBJECTS:cudart_common>
    )
    set_target_properties(${CUDART_TARGET} PROPERTIES CUDA_ARCHITECTURES "${CUDA_GEN_CODE}")
    target_include_directories(${CUDART_TARGET} PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${XSCHED_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
    )
    target_compile_options(${CUDART_TARGET} PRIVATE
        -Wall
        -Wextra
        -Wno-deprecated-gpu-targets
    )
    target_link_libraries(${CUDART_TARGET} PRIVATE
        ${CUDA_LIBRARIES}
        utils
        preempt
        halcuda
    )
    set_target_properties(${CUDART_TARGET} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    install(TARGETS ${CUDART_TARGET} RUNTIME DESTINATION bin/cuda)
endforeach()
