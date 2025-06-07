find_package(X11)

if(NOT X11_FOUND)
    return()
endif()

file(GLOB_RECURSE X11_LAUNCHER_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_executable(x11_launcher ${X11_LAUNCHER_SRCS})

target_include_directories(x11_launcher PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${X11_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(x11_launcher PRIVATE
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(x11_launcher PRIVATE
    ${X11_LIBRARIES}
    CLI11
    httplib
    service_common
    utils
    protocol
)

set_target_properties(x11_launcher PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS x11_launcher RUNTIME DESTINATION bin)
