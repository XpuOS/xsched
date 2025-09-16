if(WIN32)
    set(LEVELZERO_INCLUDE_PATHS
        $ENV{LEVELZERO_ROOT}/include
        $ENV{SystemRoot}/system32
        $ENV{windir}/system32
    )
    set(LEVELZERO_LIBRARY_PATHS
        $ENV{LEVELZERO_ROOT}/lib
        $ENV{SystemRoot}/system32
        $ENV{windir}/system32
    )
else()
    set(LEVELZERO_INCLUDE_PATHS
        $ENV{LEVELZERO_ROOT}/include
        /usr/include
        /usr/local/include
    )
    set(LEVELZERO_LIBRARY_PATHS
        $ENV{LEVELZERO_ROOT}/lib
        /usr/lib
        /usr/local/lib
    )
endif()

find_path(LEVELZERO_INCLUDE
    NAMES level_zero/ze_api.h
    PATHS ${LEVELZERO_INCLUDE_PATHS}
    DOC "Path to Level Zero include directory"
)

if(WIN32)
    find_library(LEVELZERO_LIBRARY
        NAMES ze_loader.lib ze_loader.dll
        PATHS ${LEVELZERO_LIBRARY_PATHS}
    )
else()
    find_library(LEVELZERO_LIBRARY
        NAMES ze_loader.so
        PATHS ${LEVELZERO_LIBRARY_PATHS}
    )
endif()

if(NOT LEVELZERO_INCLUDE)
    message(FATAL_ERROR "LevelZero include directory not found. Please install LevelZero SDK or set LEVELZERO_ROOT environment variable.")
endif()

if(NOT LEVELZERO_LIBRARY)
    message(FATAL_ERROR "LevelZero library not found. Please install LevelZero SDK or set LEVELZERO_ROOT environment variable.")
endif()

message(STATUS "Found LevelZero include dir: ${LEVELZERO_INCLUDE}")
message(STATUS "Found LevelZero library: ${LEVELZERO_LIBRARY}")

# levelzero common library
file(GLOB_RECURSE LEVELZERO_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(levelzero_common OBJECT ${LEVELZERO_COMMON_SRCS})
target_include_directories(levelzero_common PRIVATE
    ${LEVELZERO_INCLUDE}
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(levelzero_common PRIVATE
    ${LEVELZERO_LIBRARY}
)

# levelzero main executables
file(GLOB_RECURSE LEVELZERO_MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)

foreach(LEVELZERO_MAIN_SRC ${LEVELZERO_MAIN_SRCS})
    get_filename_component(LEVELZERO_TARGET ${LEVELZERO_MAIN_SRC} NAME_WE)
    
    add_executable(${LEVELZERO_TARGET}
        ${LEVELZERO_MAIN_SRC}
        $<TARGET_OBJECTS:levelzero_common>
    )
    
    target_include_directories(${LEVELZERO_TARGET} PRIVATE
        ${LEVELZERO_INCLUDE}
        ${CMAKE_CURRENT_LIST_DIR}/include
    )
    
    target_link_libraries(${LEVELZERO_TARGET} PRIVATE
        ${LEVELZERO_LIBRARY}
    )
    
    set_target_properties(${LEVELZERO_TARGET} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    
    install(TARGETS ${LEVELZERO_TARGET} RUNTIME DESTINATION bin/levelzero)
endforeach()
