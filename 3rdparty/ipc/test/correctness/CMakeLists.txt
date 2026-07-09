project(ipc-test-correctness)

file(GLOB TEST_CORRECT_SCRS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${TEST_CORRECT_SCRS})

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBIPC_INCLUDE_DIR})
if(MSVC)
    # MSVC uses built-in thread support and doesn't require pthread
    target_link_libraries(${PROJECT_NAME} gtest ipc)
else()
    target_link_libraries(${PROJECT_NAME} gtest ipc pthread)
endif()

install(TARGETS ${PROJECT_NAME})
