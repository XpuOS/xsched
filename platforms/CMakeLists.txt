function(add_platform platform_name)
    if(NOT platform_name)
        return()
    endif()
    
    string(TOUPPER "${platform_name}" PLATFORM_NAME)
    option(PLATFORM_${PLATFORM_NAME} "Enable support for ${PLATFORM_NAME} platform." OFF)
    if(PLATFORM_${PLATFORM_NAME})
        message(STATUS "Building platform: ${platform_name}")
        add_subdirectory(${platform_name})
    endif()
endfunction()

function(add_platform_test)
    if(BUILD_TEST)
        add_subdirectory(test)
    endif()
endfunction()

# adds a shared hal library named hal${platform_name}
function(add_hal_lib platform_name)
    cmake_parse_arguments(HAL "" "" "COMPILE_OPTIONS" ${ARGN})
    string(TOUPPER "${platform_name}" PLATFORM_NAME)
    set(SOURCE_NAME HAL_${PLATFORM_NAME}_SRCS)
    set(TARGET_NAME hal${platform_name})

    file(GLOB_RECURSE ${SOURCE_NAME}
        ${CMAKE_CURRENT_LIST_DIR}/hal/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/hal/src/*.cpp
    )
    add_library(${TARGET_NAME} SHARED ${${SOURCE_NAME}})
    add_library(XSched::${TARGET_NAME} ALIAS ${TARGET_NAME})
    target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${XSCHED_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hal/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        -fPIC
        -Wall
        -Wextra
        -Werror
        ${HAL_COMPILE_OPTIONS}
    )
    target_link_libraries(${TARGET_NAME}
        PRIVATE utils
        PRIVATE protocol
        PUBLIC  preempt
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib)
    # find hal.h
    file(GLOB_RECURSE HAL_HEADERS ${CMAKE_CURRENT_LIST_DIR}/hal/include/xsched/*/hal.h)
    foreach(HAL_HEADER ${HAL_HEADERS})
        install(FILES ${HAL_HEADER} DESTINATION include/xsched/${platform_name}/)
    endforeach()
endfunction()

# adds a shared shim library named shim${platform_name}
function(add_shim_lib platform_name)
    cmake_parse_arguments(SHIM "" "" "COMPILE_OPTIONS;SOFTLINKS" ${ARGN})
    string(TOUPPER "${platform_name}" PLATFORM_NAME)
    set(SOURCE_NAME SHIM_${PLATFORM_NAME}_SRCS)
    set(TARGET_NAME shim${platform_name})
    set(HAL_TARGET_NAME hal${platform_name})

    file(GLOB_RECURSE ${SOURCE_NAME}
        ${CMAKE_CURRENT_LIST_DIR}/shim/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/shim/src/*.cpp
    )
    add_library(${TARGET_NAME} SHARED ${${SOURCE_NAME}})
    add_library(XSched::${TARGET_NAME} ALIAS ${TARGET_NAME})
    target_include_directories(${TARGET_NAME} PUBLIC
        ${XSCHED_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/shim/include
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        -fPIC
        -Wall
        -Wextra
        -Werror
        -fvisibility=hidden
        ${SHIM_COMPILE_OPTIONS}
    )
    target_link_libraries(${TARGET_NAME}
        PRIVATE utils
        PRIVATE protocol
        PRIVATE ${HAL_TARGET_NAME}
        PUBLIC  preempt
    )
    target_link_options(${TARGET_NAME} PRIVATE
        -Wl,--exclude-libs,ALL
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib)

    # check create softlink option and the softlink list is not empty
    if(SHIM_SOFTLINK AND SHIM_SOFTLINKS)
        foreach(softlink ${SHIM_SOFTLINKS})
            install(CODE "
                execute_process(
                    COMMAND ln -s libshim${platform_name}.so ${softlink}
                    WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
                )
                message(\"-- Installing: shim softlink created: ${softlink}\")
            ")
        endforeach()
    endif()
endfunction()

# Auto-discover all platform directories
file(GLOB platform_dirs RELATIVE ${CMAKE_CURRENT_LIST_DIR} "*")
foreach(dir ${platform_dirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${dir})
        add_platform(${dir})
    endif()
endforeach()
