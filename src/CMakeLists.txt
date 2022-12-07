add_executable(lyapunov_sample
    lyapunov.cc
    )
target_link_libraries(lyapunov_sample openblas pthread gfortran)

add_executable(mlp_mnist
    sample/mlp_mnist.cc
    )
target_link_libraries(mlp_mnist openblas pthread)

add_executable(rnn_mnist
    sample/rnn_mnist.cc
    )
target_link_libraries(rnn_mnist openblas pthread)