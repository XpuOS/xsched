add_hal_lib(lg200)
set_target_properties(hallg200 PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(hallg200 PRIVATE stdc++fs)
endif()

add_executable(lg200_info tools/lg200_info.cpp)
target_include_directories(lg200_info PRIVATE
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/hal/include
)
target_compile_options(lg200_info PRIVATE -Wall -Wextra -Werror)
target_link_libraries(lg200_info PRIVATE hallg200)
install(TARGETS lg200_info RUNTIME DESTINATION bin/lg200)
set_target_properties(lg200_info PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

add_platform_test()
