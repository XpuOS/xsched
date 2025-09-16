add_hal_lib(opencl
    COMPILE_OPTIONS "-DCL_TARGET_OPENCL_VERSION=300"
)
add_shim_lib(opencl
    COMPILE_OPTIONS "-DCL_TARGET_OPENCL_VERSION=300"
    SOFTLINKS_LINUX "libOpenCL.so" "libOpenCL.so.1" "libOpenCL.so.1.0.0"
    TARGET_NAME_WINDOWS "OpenCL"
)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/shim/include/xsched/opencl/cmdq.h DESTINATION include/xsched/opencl)
add_platform_test()
