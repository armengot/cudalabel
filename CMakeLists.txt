cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)

project(cudalabel VERSION 0.01.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)

find_package(OpenCV REQUIRED)

set(CUDA_CCL ${CMAKE_CURRENT_SOURCE_DIR}/CUDA_CCL)
set(INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/interface)

include_directories(${CUDA_CCL})
link_directories(${CUDA_CCL})

# folkev's code
set(FOLKEV_SOURCES ${CUDA_CCL}/main.cu ${CUDA_CCL}/CCL.cu)
add_executable(folkev ${FOLKEV_SOURCES})
target_link_libraries(folkev PRIVATE ${OpenCV_LIBS})

# folkev's code as lib
set(FOLKEV_LIB_SOURCES ${CUDA_CCL}/CCL.cu)
include_directories(${OpenCV_INCLUDE_DIRS})
add_library(folkevlib SHARED ${FOLKEV_LIB_SOURCES})
target_include_directories(folkevlib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_options(folkevlib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)

# interface
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/interface)
set_source_files_properties(interface/cudalabel.cpp PROPERTIES LANGUAGE CUDA)

# class as lib
add_library(labelib SHARED interface/cudalabel.cpp)
set_target_properties(labelib PROPERTIES CUDA_ARCHITECTURES "60;61;62;70;72;75;80;86")
target_compile_options(labelib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)
target_link_libraries(labelib PRIVATE folkevlib ${OpenCV_LIBS})

# example of main.cpp
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;61;62;70;72;75;80;86")
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>: -O2 -Wall>) 
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE labelib folkevlib ${OpenCV_LIBS})

# Timing executable for processing images
add_executable(timing timing.cpp)
target_compile_options(timing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)
target_compile_options(timing PUBLIC $<$<COMPILE_LANGUAGE:CXX>: -O2 -Wall>)
target_include_directories(timing PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(timing PRIVATE labelib folkevlib ${OpenCV_LIBS})

