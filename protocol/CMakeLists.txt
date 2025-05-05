file(GLOB_RECURSE PROTOCOL_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(protocol STATIC ${PROTOCOL_SRCS})
add_library(XSched::protocol ALIAS protocol)

target_include_directories(protocol PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(protocol PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)
