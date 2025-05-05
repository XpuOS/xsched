add_hal_lib(hip)
add_shim_lib(hip)

if(SHIM_SOFTLINK)

    set(SHIM_HIP_SOFTLINKS
        libamdhip64.so.1
        libamdhip64.so
    )

    foreach(SOFTLINK ${SHIM_HIP_SOFTLINKS})
        install(CODE "
            execute_process(
                COMMAND ln -s libshimhip.so ${SOFTLINK}
                WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
            )
            message(\"-- Installing: softlink created: ${SOFTLINK}\")
        ")
    endforeach()

endif()

install(FILES
    hal/include/xsched/xsched_hip.py
    DESTINATION include
)