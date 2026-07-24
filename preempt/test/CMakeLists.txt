add_executable(preempt_level_capability_test
    preempt_level_capability_test.cpp
)
target_compile_options(preempt_level_capability_test PRIVATE
    -Wall
    -Wextra
    -Werror
)
target_link_libraries(preempt_level_capability_test PRIVATE
    preempt
    utils
)
add_test(NAME preempt_level_capability_test
         COMMAND preempt_level_capability_test)
