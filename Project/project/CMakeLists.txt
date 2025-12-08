cmake_minimum_required (VERSION 3.20)

# set(CMAKE_BUILD_TYPE Debug) # Do not set if profiling using NSight
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES "80;86")

project (ece408project LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

include_directories(
    "${PROJECT_SOURCE_DIR}/third_party/eigen"
    "${PROJECT_SOURCE_DIR}/src/layer/custom"
)

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_library(ece408net ece408net.cc ece408net.h)

# new-forward.cu is unused in CPU convolution, but we need to include it to make compiler happy
add_executable(m1_cpu m1_cpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/new-forward.cu")
target_link_libraries(m1_cpu ece408net MiniDNNLib)
# Set compiler flags for debug builds
target_compile_options(m1_cpu PRIVATE "$<$<CONFIG:DEBUG>:-gdwarf-5>")

add_executable(m1_gpu m1_gpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/new-forward.cu")
target_link_libraries(m1_gpu ece408net MiniDNNLib)

add_executable(m2_unroll m1_gpu.cc
                         "${PROJECT_SOURCE_DIR}/src/layer/custom/unroll-new-forward.cu"
                         "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu")
target_link_libraries(m2_unroll ece408net MiniDNNLib)

add_executable(m2_fused m1_gpu.cc
                         "${PROJECT_SOURCE_DIR}/src/layer/custom/kernel-fusion-forward.cu")
target_link_libraries(m2_fused ece408net MiniDNNLib)

add_executable(m3 m1_gpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/m3-forward.cu"
                            "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu")
target_link_libraries(m3 ece408net MiniDNNLib cublas)

# --- Milestone 3 variant executables ---

# op_0
add_executable(m3_op0
    m1_gpu.cc
    "${PROJECT_SOURCE_DIR}/m3/op_0/m3-forward.cu"
    "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu"
)
target_link_libraries(m3_op0 ece408net MiniDNNLib cublas)

# op_1
add_executable(m3_op1
    m1_gpu.cc
    "${PROJECT_SOURCE_DIR}/m3/op_1/m3-forward.cu"
    "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu"
)
target_link_libraries(m3_op1 ece408net MiniDNNLib cublas)

# op_2
add_executable(m3_op2
    m1_gpu.cc
    "${PROJECT_SOURCE_DIR}/m3/op_2/m3-forward.cu"
    "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu"
)
target_link_libraries(m3_op2 ece408net MiniDNNLib cublas)

# op_4
add_executable(m3_op4
    m1_gpu.cc
    "${PROJECT_SOURCE_DIR}/m3/op_4/m3-forward.cu"
    "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu"
)
target_link_libraries(m3_op4 ece408net MiniDNNLib cublas)

# op_6
add_executable(m3_op6
    m1_gpu.cc
    "${PROJECT_SOURCE_DIR}/m3/op_6/m3-forward.cu"
    "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu"
)
target_link_libraries(m3_op6 ece408net MiniDNNLib cublas)

# req_0
add_executable(m3_req0
    m1_gpu.cc
    "${PROJECT_SOURCE_DIR}/m3/req_0/m3-forward.cu"
    "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu"
)
target_link_libraries(m3_req0 ece408net MiniDNNLib cublas)

# req_1
add_executable(m3_req1
    m1_gpu.cc
    "${PROJECT_SOURCE_DIR}/m3/req_1/m3-forward.cu"
    "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul.cu"
)
target_link_libraries(m3_req1 ece408net MiniDNNLib cublas)

