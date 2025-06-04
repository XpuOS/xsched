add_hal_lib(cuda LINK_LIBRARIES cuxtra)
add_shim_lib(cuda SOFTLINKS libcuda.so.1 libcuda.so)
add_platform_test()
