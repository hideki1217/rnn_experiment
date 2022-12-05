add_executable(ryapunov_sample
    ryapunov.cc
    )
target_link_libraries(ryapunov_sample openblas pthread gfortran)

add_executable(mlp_mnist
    sample/mlp_mnist.cc
    )
target_link_libraries(mlp_mnist openblas pthread)

add_executable(rnn_mnist
    sample/rnn_mnist.cc
    )
target_link_libraries(rnn_mnist openblas pthread)

add_executable(exp0
    exp0.cc
    )
target_link_libraries(exp0 openblas pthread gfortran)