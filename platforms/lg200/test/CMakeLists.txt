if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    return()
endif()

add_executable(lg200_resource_test lg200_resource_test.cpp)
target_compile_options(lg200_resource_test PRIVATE -Wall -Wextra -Werror)
target_link_libraries(lg200_resource_test PRIVATE hallg200)
add_test(NAME lg200_resource_test COMMAND lg200_resource_test)
set_tests_properties(lg200_resource_test PROPERTIES TIMEOUT 30)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(lg200_resource_test PRIVATE stdc++fs)
endif()
