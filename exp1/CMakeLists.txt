add_executable(exp1
    main.cc
    )
target_link_libraries(exp1 openblas pthread gfortran)
target_include_directories(exp1 PRIVATE /home/okumura/workspace/rnn_experiment/src)