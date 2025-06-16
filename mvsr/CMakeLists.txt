cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_VERSION 20)
set(WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

if (MVSC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

project(mvsr)

add_library(mvsr SHARED
    src/mvsr.cpp
    inc/mvsr.h
)
target_include_directories(mvsr PRIVATE inc)

include(CTest)
add_executable(TestLinear test/linear.cpp)
target_link_libraries(TestLinear PUBLIC mvsr)
add_test(NAME TestLinear COMMAND TestLinear)

