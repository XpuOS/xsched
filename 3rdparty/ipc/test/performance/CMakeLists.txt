project(ipc-test-performance)

# Server
set(SERVER ipc-test-performance-server)
add_executable(${SERVER} test_server.cpp)
target_include_directories(${SERVER} PRIVATE ${LIBIPC_INCLUDE_DIR})
target_link_libraries(${SERVER} PRIVATE ipc)
install(TARGETS ${SERVER} RUNTIME DESTINATION bin)

# Client
set(CLIENT ipc-test-performance-client)
add_executable(${CLIENT} test_client.cpp)
target_include_directories(${CLIENT} PRIVATE ${LIBIPC_INCLUDE_DIR})
target_link_libraries(${CLIENT} PRIVATE ipc)
install(TARGETS ${CLIENT} RUNTIME DESTINATION bin)
