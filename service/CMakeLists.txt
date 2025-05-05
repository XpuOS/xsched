# xcli
file(GLOB_RECURSE XCLI_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/cli/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/cli/src/*.cpp
)
add_executable(xcli ${XCLI_SRCS})

target_include_directories(xcli PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/common
    ${CMAKE_CURRENT_LIST_DIR}/cli/include
)

target_compile_options(xcli PRIVATE
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(xcli PRIVATE
    ipc
    CLI11
    httplib
    tabulate
    jsoncpp_static
    utils
    sched
    protocol
)

set_target_properties(xcli PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS xcli RUNTIME DESTINATION bin)

# xserver
file(GLOB_RECURSE XSERVER_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/server/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/server/src/*.cpp
)
add_executable(xserver ${XSERVER_SRCS})

target_include_directories(xserver PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/common
    ${CMAKE_CURRENT_LIST_DIR}/server/include
)

target_compile_options(xserver PRIVATE
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(xserver PRIVATE
    ipc
    httplib
    jsoncpp_static
    utils
    sched
    protocol
)

set_target_properties(xserver PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS xserver RUNTIME DESTINATION bin)
