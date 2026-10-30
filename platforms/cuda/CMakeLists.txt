add_hal_lib(cuda LINK_LIBRARIES cuxtra)
add_shim_lib(cuda
    SOFTLINKS_LINUX libcuda.so.1 libcuda.so
    TARGET_NAME_WINDOWS "nvcuda"
)
add_platform_test()

# Optional preload helper for experimental Linux RM channel control.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_library(cuda_channel_tracker SHARED tools/channel_tracker.cpp)
    target_include_directories(cuda_channel_tracker PRIVATE
        ${PROJECT_SOURCE_DIR}/utils/include ${CMAKE_CURRENT_SOURCE_DIR}/hal/include)
    target_compile_options(cuda_channel_tracker PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(cuda_channel_tracker PRIVATE pthread ${CMAKE_DL_LIBS})
    install(TARGETS cuda_channel_tracker LIBRARY DESTINATION lib)
    if(BUILD_TEST)
        add_executable(cuda_channel_tracker_test test/unit/channel_tracker.cpp)
        target_include_directories(cuda_channel_tracker_test PRIVATE
            ${PROJECT_SOURCE_DIR}/utils/include ${CMAKE_CURRENT_SOURCE_DIR}/hal/include)
        target_compile_options(cuda_channel_tracker_test PRIVATE -Wall -Wextra -Werror)
        target_link_libraries(cuda_channel_tracker_test PRIVATE pthread ${CMAKE_DL_LIBS})
        add_test(NAME cuda_channel_tracker_test COMMAND cuda_channel_tracker_test)
    endif()
endif()
