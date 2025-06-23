cmake_minimum_required(VERSION 3.16)
project(pwreg)
set(CMAKE_CXX_STANDARD 20)

add_library(eigen INTERFACE)
target_include_directories(eigen
    INTERFACE "eigen/Eigen"
)


add_library(pwreg SHARED
    polyreg.cpp
)
target_link_libraries(pwreg
    PRIVATE eigen
)
target_compile_definitions(pwreg
    PRIVATE EIGEN_STACK_ALLOCATION_LIMIT=0
)
target_compile_options(pwreg PUBLIC
    -Wno-changes-meaning # used because redeclaration (template specialization) of "Part" within regression
)

add_executable(heap_test
    heap_test.cpp
)
add_executable(list_test
    list_test.cpp
)

add_executable(polyreg_test
    polyreg_test.cpp
)
target_link_libraries(polyreg_test
    pwreg
)
