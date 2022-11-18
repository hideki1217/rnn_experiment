add_library(mynn STATIC
    mynn.cc
    mynn.h
)
target_link_libraries(mynn openblas)


add_executable(mynn_sample
    main.cc
)
target_link_libraries(mynn_sample mynn)

add_executable(mnist
    mnist.cc
)
target_link_libraries(mnist mynn)

add_executable(rnn_classify
    rnn_classify.cc
    )
target_link_libraries(rnn_classify mynn)
