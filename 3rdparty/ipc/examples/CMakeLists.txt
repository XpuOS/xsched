cmake_minimum_required(VERSION 3.12)

project(example)

set(IPC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../output/include")
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(IPC_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../output/lib/libipc.a")
else()
    set(IPC_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../output/lib/ipc.lib")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

if(MSVC)
    # Overrides the default Debug compilation options of CMake,
    # avoiding conflicts between /MTd(ipc) and /MDd(gtest).
    set(CMAKE_C_FLAGS_DEBUG   "$ENV{CFLAGS}   /Od /Zi")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} /Od /Zi")
endif()

add_executable(sender ${CMAKE_CURRENT_SOURCE_DIR}/sender.cpp)
add_executable(receiver ${CMAKE_CURRENT_SOURCE_DIR}/receiver.cpp)

# Add the IPC library
add_library(ipc STATIC IMPORTED)
set_target_properties(ipc PROPERTIES
    IMPORTED_LOCATION "${IPC_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${IPC_INCLUDE_DIR}"
)

target_link_libraries(sender ipc)
target_link_libraries(receiver ipc)
target_include_directories(sender PRIVATE ${IPC_INCLUDE_DIR})
target_include_directories(receiver PRIVATE ${IPC_INCLUDE_DIR})

install(TARGETS sender receiver
    RUNTIME DESTINATION bin
)