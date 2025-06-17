cmake_minimum_required(VERSION 3.10)
project(yolov8_byte_tracker)

set(CMAKE_CXX_STANDARD 14)


# Mac下libomp支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")

find_package(OpenCV REQUIRED)

# ncnn
set(NCNN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/ncnn-20240410-macos-vulkan/ncnn.framework/Versions/A/Headers)
find_package(ncnn REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ncnn_INCLUDE_DIRS}
    byte_tracker/include
    yolov8
    yolov11
    common
)

add_subdirectory(byte_tracker)
add_subdirectory(common)
add_subdirectory(yolov8)
add_subdirectory(yolov11)

add_executable(yolo_tracking main.cpp)

# 默认链接 yolov8_lib，如需切换 yolov11，修改 main.cpp include 和此处链接库
# target_link_libraries(yolo_tracking yolov8_lib)
target_link_libraries(yolo_tracking
    ${OpenCV_LIBS}
    ncnn
    byte_tracker
    common
    yolov8_lib
) 