cmake_minimum_required(VERSION 3.12)

project(XSched VERSION 1.0.0)

option(SHIM_SOFTLINK     "Create softlink for shim library."     OFF)
option(CONTAINER_SUPPORT "Enable support for container runtime." OFF)
option(BUILD_TEST        "Build test cases."                     OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS_RELEASE   "$ENV{CFLAGS}   -O3 -DRELEASE_MODE")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DRELEASE_MODE")
set(CMAKE_C_FLAGS_DEBUG     "$ENV{CFLAGS}   -O0 -g2 -ggdb -DDEBUG_MODE")
set(CMAKE_CXX_FLAGS_DEBUG   "$ENV{CXXFLAGS} -O0 -g2 -ggdb -DDEBUG_MODE")
set(XSCHED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(3rdparty)
add_subdirectory(utils)
add_subdirectory(protocol)
add_subdirectory(sched)
add_subdirectory(preempt)
add_subdirectory(service)
add_subdirectory(platforms)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/XSchedConfig.cmake
    INSTALL_DESTINATION lib/cmake/XSched
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/XSchedConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/XSchedConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/XSchedConfigVersion.cmake
    DESTINATION lib/cmake/XSched
)

export(EXPORT XSchedTargets FILE ${CMAKE_CURRENT_BINARY_DIR}/XSchedTargets.cmake)
install(EXPORT XSchedTargets NAMESPACE XSched:: DESTINATION lib/cmake/XSched)
install(DIRECTORY ${XSCHED_INCLUDE_DIR}/ DESTINATION include)
