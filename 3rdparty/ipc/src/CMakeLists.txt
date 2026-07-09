# ipc library
file(GLOB_RECURSE IPC_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)
add_library(ipc STATIC ${IPC_SRCS})

target_include_directories(ipc PUBLIC
    ${LIBIPC_INCLUDE_DIR}
    ${LIBIPC_TOOLS_DIR}
)

target_link_libraries(ipc PUBLIC
    Boost::interprocess
)

# Enabling PIC allows it to be directly used when linked to dynamic libraries
set_property(TARGET ipc PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS ipc)