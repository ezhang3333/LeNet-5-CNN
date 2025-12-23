#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel_restrict(const float * __restrict__ input,
                                             const float * __restrict__ mask,
                                             float * __restrict__ output,
                                             int Batch, int Map_out, int Channel,
                                             int Height, int Width, int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= Batch || h_out >= H_out || w_out >= W_out) {
        return;
    }

    for (int m = 0; m < Map_out; m++) {
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    int h_in = h_out + p;
                    int w_in = w_out + q;

                    size_t in_idx =
                        (((size_t)b * Channel + c) * Height + h_in) * Width + w_in;

                    size_t mask_idx =
                        (((size_t)m * Channel + c) * K + p) * K + q;

                    acc += input[in_idx] * mask[mask_idx];
                }
            }
        }

        size_t out_idx =
            (((size_t)b * Map_out + m) * H_out + h_out) * W_out + w_out;

        output[out_idx] = acc;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float * /*host_output*/,
                                                    const float * host_input,
                                                    const float * host_mask,
                                                    float **device_output_ptr,
                                                    float **device_input_ptr,
                                                    float **device_mask_ptr,
                                                    const int Batch, const int Map_out,
                                                    const int Channel, const int Height,
                                                    const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    size_t in_elems = (size_t)Batch * Channel * Height * Width;
    size_t out_elems = (size_t)Batch * Map_out * H_out * W_out;
    size_t m_elems = (size_t)Map_out * Channel * K * K;

    size_t in_bytes = in_elems * sizeof(float);
    size_t out_bytes = out_elems * sizeof(float);
    size_t m_bytes = m_elems * sizeof(float);

    cudaMalloc((void **)device_input_ptr, in_bytes);
    cudaMalloc((void **)device_output_ptr, out_bytes);
    cudaMalloc((void **)device_mask_ptr, m_bytes);

    cudaMemcpy(*device_input_ptr, host_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, m_bytes, cudaMemcpyHostToDevice);

    cudaMemset(*device_output_ptr, 0, out_bytes);
}

__host__ void GPUInterface::conv_forward_gpu(float * device_output,
                                             const float * device_input,
                                             const float * device_mask,
                                             const int Batch, const int Map_out,
                                             const int Channel, const int Height,
                                             const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid((W_out + TILE_WIDTH - 1) / TILE_WIDTH,
              (H_out + TILE_WIDTH - 1) / TILE_WIDTH,
              Batch);

    conv_forward_kernel_restrict<<<grid, block>>>(
        device_input,
        device_mask,
        device_output,
        Batch, Map_out, Channel, Height, Width, K
    );

    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float * host_output,
                                                    float * device_output,
                                                    float * device_input,
                                                    float * device_mask,
                                                    const int Batch, const int Map_out,
                                                    const int Channel, const int Height,
                                                    const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    size_t out_elems = (size_t)Batch * Map_out * H_out * W_out;
    size_t out_bytes = out_elems * sizeof(float);

    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "
                 <<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "
                 <<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}