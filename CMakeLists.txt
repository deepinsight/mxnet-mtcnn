# cmake verson，指定cmake的最小版本号
cmake_minimum_required(VERSION 3.2)

# project name，指定项目的名称，一般和项目的文件夹名称对应
project(face_detect C CXX)

# set environment variable，设置环境变量
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_INCLUDE_DIRS: ${OpenMP_INCLUDE_DIRS}")
    message(STATUS "OpenMP_LIBS: ${OpenMP_LIBS}")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
endif()

add_subdirectory(src)
