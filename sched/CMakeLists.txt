file(GLOB_RECURSE SCHED_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(sched STATIC ${SCHED_SRCS})
add_library(XSched::sched ALIAS sched)

target_include_directories(sched PUBLIC
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(sched PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(sched PRIVATE
    ipc
    utils
    protocol
)
