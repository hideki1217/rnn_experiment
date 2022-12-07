include_directories(../src)

add_executable(test_lyapunov test_lyapunov.cc)
target_link_libraries(test_lyapunov openblas pthread)

