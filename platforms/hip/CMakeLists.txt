add_hal_lib(hip)
add_shim_lib(hip
    SOFTLINKS_LINUX libamdhip64.so.1 libamdhip64.so
)

target_include_directories(shimhip PRIVATE
    ${CMAKE_SOURCE_DIR}/sched/include
    ${CMAKE_SOURCE_DIR}/preempt/include
)

install(FILES
    hal/include/xsched/xsched_hip.py
    DESTINATION include
)
