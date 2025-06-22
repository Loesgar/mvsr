cmake_minimum_required(VERSION 3.15)

project(mvsr
    HOMEPAGE_URL https://github.com/Loesgar/mvsr
)

set(CMAKE_CXX_VERSION 20)
set(WINDOWS_EXPORT_ALL_SYMBOLS ON)

if (MVSC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

add_library(mvsr SHARED
    src/mvsr.cpp
    inc/mvsr.h
)
target_include_directories(mvsr PRIVATE inc)
target_link_options(mvsr PRIVATE -static-libgcc -static-libstdc++)

set_target_properties(mvsr PROPERTIES PUBLIC_HEADER
    inc/mvsr.h
)
install(TARGETS mvsr
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION inc
)

option(TESTING "Enable tests." ON)
if (TESTING)
    include(CTest)
    add_executable(test-linear test/linear.cpp)
    target_link_libraries(test-linear PUBLIC mvsr)
    add_test(NAME test-linear COMMAND test-linear)
endif()
