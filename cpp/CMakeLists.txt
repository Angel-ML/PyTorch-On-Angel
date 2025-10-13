cmake_minimum_required(VERSION 3.12)
project(torch_on_angel)

set(CMAKE_CXX_STANDARD 17)

#set libtorch path
#set(TORCH_HOME "/path/to/libtorch")
set(TORCH_HOME $ENV{TORCH_HOME})
set(CMAKE_PREFIX_PATH ${TORCH_HOME})

find_package(Torch)
if (TORCH_FOUND)
    if(Torch_VERSION STRGREATER_EQUAL 1.4.0)
        add_definitions(-DLIBTORCH_VERSION_LATEST)
    endif()
    message("-- FOUND TORCH VERSION : ${Torch_VERSION}")
    if(Torch_VERSION VERSION_GREATER_EQUAL 1.13.1)
            add_definitions(-DLIBTORCH_VERSION_GREATER_1_13_1)
            message("-- FOUND TORCH VERSION : ${Torch_VERSION}, add ${-DLIBTORCH_VERSION_GREATER_1_13_1}")
    endif()
endif (TORCH_FOUND)
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
add_subdirectory(pytorch_scatter-2.1.2)
add_library(torch_angel SHARED ${PYTORCH_SRC} ${PYTORCH_HEADER})
target_link_libraries(torch_angel ${TORCH_LIBRARIES})
target_link_libraries(torch_angel -Wl,--no-as-needed torchscatter)