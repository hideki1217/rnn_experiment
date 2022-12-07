add_executable(exp0
    main.cc
    )
target_link_libraries(exp0 openblas pthread gfortran)
target_include_directories(exp0 PRIVATE /home/okumura/workspace/rnn_experiment/src)