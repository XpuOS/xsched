file(GLOB_RECURSE PREEMPT_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(preempt SHARED ${PREEMPT_SRCS})
add_library(XSched::preempt ALIAS preempt)

target_include_directories(preempt PUBLIC
    $<BUILD_INTERFACE:${XSCHED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(preempt PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(preempt
    PRIVATE utils
    PRIVATE sched
    PRIVATE protocol
)

set_target_properties(preempt PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS preempt EXPORT XSchedTargets LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include)
