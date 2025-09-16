add_hal_lib(hip)
add_shim_lib(hip
    SOFTLINKS_LINUX libamdhip64.so.1 libamdhip64.so
)

install(FILES
    hal/include/xsched/xsched_hip.py
    DESTINATION include
)
