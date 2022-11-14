cmake_minimum_required(VERSION 3.22)

project(mynn)

set_property(GLOBAL PROPERTY CXX_STANDARD 17)

link_directories(/usr/local/OpenBLAS/lib)
include_directories(/usr/local/OpenBLAS/include/openblas)

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
