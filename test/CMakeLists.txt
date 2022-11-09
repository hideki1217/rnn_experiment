include_directories(../src)

add_executable(test_linear
    test_linear.cc
)

add_test(NAME test_linear COMMAND test_linear)
