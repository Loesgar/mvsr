cmake_minimum_required(VERSION 3.15)

project(mvsr
    DESCRIPTION "Library for Fast Segmented Regression"
    HOMEPAGE_URL https://github.com/Loesgar/mvsr
    VERSION 0.2.0
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(mvsr SHARED
    src/mvsr.cpp
    inc/mvsr.h
)
target_include_directories(mvsr PUBLIC inc)
set_property(TARGET mvsr PROPERTY PUBLIC_HEADER
    inc/mvsr.h
)

if (MSVC)
    target_compile_options(mvsr PRIVATE /W4 /WX)
    set_property(TARGET mvsr PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET mvsr PROPERTY PREFIX "lib")
else()
    target_compile_options(mvsr PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(mvsr PRIVATE -Wno-missing-field-initializers)
    if (NOT APPLE)
        target_link_options(mvsr PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()

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
