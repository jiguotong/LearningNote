cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories("/usr/include/eigen3")
include_directories("/usr/include/onnxruntime")

#Include path to header files for Custom Op Test
include_directories("./")

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /usr/lib/onnxruntime)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})