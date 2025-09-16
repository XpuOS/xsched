function(add_platform platform_name)
    if(NOT platform_name)
        return()
    endif()

    string(TOUPPER "${platform_name}" PLATFORM_NAME)
    # Create an option for each platform (disabled by default)
    # to allow users to choose whether to build the platform or not
    option(PLATFORM_${PLATFORM_NAME} "Enable support for ${PLATFORM_NAME} platform." OFF)
    # If the option is enabled, add subdirectories for the corresponding platform
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

# adds a shared(Linux)/static(Windows) hal library named hal${platform_name}
function(add_hal_lib platform_name)
    cmake_parse_arguments(HAL "" "" "COMPILE_OPTIONS;LINK_LIBRARIES" ${ARGN})
    string(TOUPPER "${platform_name}" PLATFORM_NAME)
    set(SOURCE_NAME HAL_${PLATFORM_NAME}_SRCS)
    set(TARGET_NAME hal${platform_name})

    file(GLOB_RECURSE ${SOURCE_NAME}
        ${CMAKE_CURRENT_LIST_DIR}/hal/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/hal/src/*.cpp
    )

    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        add_library(${TARGET_NAME} SHARED ${${SOURCE_NAME}})
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        add_library(${TARGET_NAME} STATIC ${${SOURCE_NAME}})
        set_target_properties(${TARGET_NAME} PROPERTIES
            SUFFIX ".dll"           # Ensure that the dynamic library suffix is .dll
            IMPORT_SUFFIX ".lib"    # Mandatory import of library suffix .lib
        )
    endif()
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

    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(${TARGET_NAME}
            PRIVATE utils
            PRIVATE protocol
            PUBLIC  preempt
            PRIVATE ${HAL_LINK_LIBRARIES}
        )
        set_target_properties(${TARGET_NAME} PROPERTIES
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
        install(TARGETS ${TARGET_NAME} EXPORT XSchedTargets LIBRARY DESTINATION lib)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_libraries(${TARGET_NAME}
            PRIVATE utils
            PRIVATE protocol
            PRIVATE preempt
            PRIVATE ${HAL_LINK_LIBRARIES}
        )
    endif()

    # find hal.h
    file(GLOB_RECURSE HAL_HEADERS ${CMAKE_CURRENT_LIST_DIR}/hal/include/xsched/*/hal.h)
    foreach(HAL_HEADER ${HAL_HEADERS})
        install(FILES ${HAL_HEADER} DESTINATION include/xsched/${platform_name}/)
    endforeach()
endfunction()

# adds a shared shim library named shim${platform_name} (on Linux) 
# or a static shim library named as specified by TARGET_NAME_WINDOWS (on Windows)
function(add_shim_lib platform_name)
    cmake_parse_arguments(SHIM "" "" "COMPILE_OPTIONS;LINK_LIBRARIES;SOFTLINKS_LINUX;TARGET_NAME_WINDOWS" ${ARGN})
    string(TOUPPER "${platform_name}" PLATFORM_NAME)
    set(SOURCE_NAME SHIM_${PLATFORM_NAME}_SRCS)
    set(HAL_TARGET_NAME hal${platform_name})
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(TARGET_NAME shim${platform_name})
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        # On Windows, the target name is set to the driver name
        # XSched needs library name to be the same as the driver to complete DLL proxy and API interception
        if("${SHIM_TARGET_NAME_WINDOWS}" STREQUAL "")
            message(FATAL_ERROR "Must specify TARGET_NAME_WINDOWS when calling add_shim_lib() on Windows.")
        endif()
        set(TARGET_NAME ${SHIM_TARGET_NAME_WINDOWS})
    endif()

    file(GLOB_RECURSE ${SOURCE_NAME}
        ${CMAKE_CURRENT_LIST_DIR}/shim/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/shim/src/*.cpp
    )
    add_library(${TARGET_NAME} SHARED ${${SOURCE_NAME}})
    add_library(XSched::${TARGET_NAME} ALIAS ${TARGET_NAME})
    target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${XSCHED_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/shim/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        -fPIC
        -Wall
        -Wextra
        -Werror
        -fvisibility=hidden 
        ${SHIM_COMPILE_OPTIONS}
    )

    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(${TARGET_NAME}
            PRIVATE utils
            PRIVATE protocol
            PRIVATE ${HAL_TARGET_NAME}
            PUBLIC  preempt
            PRIVATE ${SHIM_LINK_LIBRARIES}
        )
        # create softlinks
        if(SHIM_SOFTLINK)
            foreach(softlink ${SHIM_SOFTLINKS_LINUX})
                install(CODE "
                    execute_process(
                        COMMAND ln -s libshim${platform_name}.so ${softlink}
                        WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
                    )
                    message(\"-- Installing: shim softlink created: ${softlink}\")
                ")
            endforeach()
        endif()
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_link_libraries(${TARGET_NAME}
            PRIVATE utils
            PRIVATE protocol
            PRIVATE ${HAL_TARGET_NAME}
            PRIVATE preempt
            PRIVATE ${SHIM_LINK_LIBRARIES}
        )
        # On the Windows, the `lib` prefix and `.dll` suffix are added by default.
        # use set_target_properties to modify the output name and remove the `lib` prefix
        set_target_properties(${TARGET_NAME} PROPERTIES
            OUTPUT_NAME "${TARGET_NAME}"
            PREFIX ""               # Remove the `lib` prefix for DLL(.dll)
            IMPORT_PREFIX ""        # Remove the `lib` prefix for import libraries(.dll.a)
            SUFFIX ".dll"           # Ensure that the dynamic library suffix is .dll
            IMPORT_SUFFIX ".lib"    # Mandatory import of library suffix .lib
        )
    endif()

    target_link_options(${TARGET_NAME} PRIVATE
        -Wl,--exclude-libs,ALL # Tell linker to exclude symbols from all static libraries
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    install(TARGETS ${TARGET_NAME} EXPORT XSchedTargets LIBRARY DESTINATION lib)
endfunction()

# Auto-discover all platform directories
file(GLOB platform_dirs RELATIVE ${CMAKE_CURRENT_LIST_DIR} "*")
foreach(dir ${platform_dirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${dir})
        add_platform(${dir})
    endif()
endforeach()
