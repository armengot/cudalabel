cmake_minimum_required(VERSION 3.0)
project(cudalabel)

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

# example of main.cpp
add_executable(${PROJECT_NAME} main.cpp interface/cudalabel.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)

# link
target_link_libraries(${PROJECT_NAME} PRIVATE folkevlib ${OpenCV_LIBS})

# Timing executable for processing images
add_executable(timing timing.cpp interface/cudalabel.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>)
target_include_directories(timing PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(timing PRIVATE folkevlib ${OpenCV_LIBS})
