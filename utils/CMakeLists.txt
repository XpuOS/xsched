file(GLOB_RECURSE UTILS_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(utils STATIC ${UTILS_SRCS})
add_library(XSched::utils ALIAS utils)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_options(utils PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)
target_link_libraries(utils PUBLIC pthread dl)
