add_library(mynn STATIC
    mynn.cc
    mynn.h
)
target_link_libraries(mynn openblas)


add_executable(mynn_sample
    main.cc
)
target_link_libraries(mynn_sample mynn)