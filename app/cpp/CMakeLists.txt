add_library(lenet_core)

target_sources(lenet_core PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/network.cc
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/layer/conv.cc
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/layer/relu.cc
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/layer/max_pooling.cc
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/layer/fully_connected.cc
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/layer/softmax.cc
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/loss/cross_entropy_loss.cc
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src/optimizer/sgd.cc
)

target_include_directories(lenet_core PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/third_party/eigen
    ${CMAKE_CURRENT_LIST_DIR}/../../cnn/src
)

add_executable(lenet_cpu
    ${CMAKE_CURRENT_LIST_DIR}/idx_dataset.cc
    ${CMAKE_CURRENT_LIST_DIR}/lenet_cpu_main.cc
)
target_link_libraries(lenet_cpu PRIVATE lenet_core)

