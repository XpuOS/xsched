add_hal_lib(cuda LINK_LIBRARIES cuxtra)
add_shim_lib(cuda
    SOFTLINKS_LINUX libcuda.so.1 libcuda.so
    TARGET_NAME_WINDOWS "nvcuda"
)

target_include_directories(shimcuda PRIVATE
    ${CMAKE_SOURCE_DIR}/sched/include
    ${CMAKE_SOURCE_DIR}/preempt/include
)

add_platform_test()
