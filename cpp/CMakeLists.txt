cmake_minimum_required(VERSION 3.12)
project(torch_on_angel)

set(CMAKE_CXX_STANDARD 14)

#set libtorch path
#set(TORCH_HOME "/path/to/libtorch")
#set(TORCH_HOME $ENV{TORCH_HOME})
set(TORCH_HOME "/Users/ouyangwen/software/libtorch-1.3/libtorch")
set(CMAKE_PREFIX_PATH ${TORCH_HOME})

find_package(Torch REQUIRED)
find_package(JNI REQUIRED)

include_directories(${JAVA_INCLUDE_PATH})
include_directories(${JAVA_INCLUDE_PATH}/linux)
include_directories("include")
include_directories(${TORCH_HOME}/include/torch/csrc/api/include)
include_directories(${TORCH_HOME}/include)

set(PYTORCH_HEADER
        include/angel/pytorch/model.h
        include/angel/pytorch/angel_torch.h
        include/angel/commons.h
        include/angel/timer.h
        include/angel/map.h
        include/angel/pytorch/utils.h)

set(PYTORCH_SRC
        src/angel/pytorch/angel_torch.cc
        src/angel/pytorch/utils.cc
        src/angel/map.cc src/angel/pytorch/model.cc)

add_library(torch_angel SHARED ${PYTORCH_SRC} ${PYTORCH_HEADER})
target_link_libraries(torch_angel ${TORCH_LIBRARIES})
