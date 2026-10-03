add_hal_lib(cuda LINK_LIBRARIES cuxtra)
add_shim_lib(cuda
    LINK_LIBRARIES cuxtra
    SOFTLINKS_LINUX libcuda.so.1 libcuda.so
    TARGET_NAME_WINDOWS "nvcuda"
)
add_platform_test()
