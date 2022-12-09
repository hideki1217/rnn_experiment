add_executable(exp0
    exp.cc
    )
target_link_libraries(exp0 openblas pthread gfortran)
target_compile_definitions(exp0 
PRIVATE EXP_NAME="exp0"
PRIVATE EXP_ID=0
    )
target_include_directories(exp0 PRIVATE /home/okumura/workspace/rnn_experiment/src)

add_executable(exp1
    exp.cc
    )
target_link_libraries(exp1 openblas pthread gfortran)
target_compile_definitions(exp1
PRIVATE EXP_NAME="exp1"
PRIVATE EXP_ID=1
    )
target_include_directories(exp1 PRIVATE /home/okumura/workspace/rnn_experiment/src)