cmake_minimum_required(VERSION 3.8)
project(GradientDescent)
set(CMAKE_CXX_STANDARD 14)
set($ARMADILLO_INCLUDE_DIRS /usr/local/lib)

#set(CMAKE_EXE_LINKER_FLAGS "-O2 -larmadillo")
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(src)
include_directories(inc)
add_executable(GradientDescent src/main.cpp inc/CSVRow.h inc/CSVIterator.h)

target_link_libraries(GradientDescent ${ARMADILLO_LIBRARIES})


set(SOURCE_FILES src/main.cpp)

