find_package(X11)

if(NOT X11_FOUND)
    return()
endif()

file(GLOB_RECURSE X11_MONITOR_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_executable(x11_monitor ${X11_MONITOR_SRCS})

target_include_directories(x11_monitor PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${X11_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(x11_monitor PRIVATE
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(x11_monitor PRIVATE
    ${X11_LIBRARIES}
    CLI11
    httplib
    service_common
    utils
    protocol
)

set_target_properties(x11_monitor PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS x11_monitor RUNTIME DESTINATION bin)
