add_hal_lib(levelzero)
add_shim_lib(levelzero
    SOFTLINKS_LINUX "libze_loader.so" "libze_loader.so.1" "libze_loader.so.1.22.1"
    TARGET_NAME_WINDOWS "ze_loader"
)
add_platform_test()
