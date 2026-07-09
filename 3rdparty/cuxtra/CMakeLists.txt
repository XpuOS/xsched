cmake_minimum_required(VERSION 3.12)

project(cuXtra)

string(TOLOWER "${CMAKE_SYSTEM_NAME}"      OS_NAME_LOWER)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_NAME_LOWER)

set(CUXTRA_LIB ${CMAKE_CURRENT_LIST_DIR}/lib/libcuxtra_${OS_NAME_LOWER}_${ARCH_NAME_LOWER}.a)
if(NOT EXISTS ${CUXTRA_LIB})
    message(FATAL_ERROR "cuXtra does not support ${CMAKE_SYSTEM_PROCESSOR} on ${CMAKE_SYSTEM_NAME}")
endif()

add_library(cuxtra INTERFACE)

target_include_directories(cuxtra INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(cuxtra INTERFACE
    ${CUXTRA_LIB}
)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(cuxtra INTERFACE dl)
elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(cuxtra INTERFACE shlwapi)
endif()

install(FILES ${CUXTRA_LIB} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/cuxtra DESTINATION include)
