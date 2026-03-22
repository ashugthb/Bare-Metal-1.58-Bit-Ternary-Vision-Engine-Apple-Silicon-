cmake_minimum_required(VERSION 3.10)
project(MacTernaryVision)

set(CMAKE_CXX_STANDARD 14)

# Homebrew OpenCV on Apple Silicon
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(MacTernaryVision main.cpp benchmark_mac.cpp)
target_link_libraries(MacTernaryVision ${OpenCV_LIBS})
