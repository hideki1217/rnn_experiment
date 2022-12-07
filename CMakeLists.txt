cmake_minimum_required(VERSION 3.0)

project(mynn)

set_property(GLOBAL PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -march=native -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -DNDEBUG -g -pg")

link_directories(~/local/OpenBLAS/lib)
include_directories(~/local/OpenBLAS/include/openblas)

enable_testing()

add_subdirectory(src)
add_subdirectory(exp0)
add_subdirectory(exp1)
add_subdirectory(test)
