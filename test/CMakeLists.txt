include_directories(../src)

add_executable(test_linear
    test_linear.cc
)
target_link_libraries(test_linear mynn)

add_test(NAME test_linear COMMAND test_linear)

