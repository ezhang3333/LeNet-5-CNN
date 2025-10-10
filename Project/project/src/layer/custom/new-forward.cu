#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "gpu-new-forward.h"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// Basic untiled (except output tiling) forward convolution kernel
// Grid mapping:
//   grid.x = Map_out (output feature maps)
//   grid.y = number of tiles per output feature map (HGrid * WGrid)
//   grid.z = Batch (images)
// Block mapping:
//   blockDim = (TILE_WIDTH, TILE_WIDTH, 1)
//   Each thread computes one (h,w) output position for the (b,m) map
__global__ void conv_forward_kernel(
    float *output,
    const float *input,
    const float *mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    // grid decomposition
    const int m = blockIdx.x;           // which output feature map
    const int b = blockIdx.z;           // which image in the batch

    // number of tiles per row for this output feature
    const int W_grid = (Width_out  + TILE_WIDTH - 1) / TILE_WIDTH;
    // const int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH; // not needed explicitly

    // location of this thread within its tile
    const int tile_linear = blockIdx.y;
    const int h0 = (tile_linear / W_grid) * TILE_WIDTH + threadIdx.y;
    const int w0 = (tile_linear % W_grid) * TILE_WIDTH + threadIdx.x;

    if (h0 < Height_out && w0 < Width_out) {
        float acc = 0.0f;
        // Sum over input channels and KxK filter elements
        for (int c = 0; c < Channel; ++c) {
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    const int in_h = h0 + p;
                    const int in_w = w0 + q;
                    // NCHW indexing
                    const float x = input[b * (Channel * Height * Width)
                                         + c * (Height * Width)
                                         + in_h * Width + in_w];
                    const float w = mask[m * (Channel * K * K)
                                         + c * (K * K)
                                         + p * K + q];
                    acc += x * w;
                }
            }
        }
        // Write output: N (b) x M (m) x H_out x W_out
        output[b * (Map_out * Height_out * Width_out)
             + m * (Height_out * Width_out)
             + h0 * Width_out + w0] = acc;
    }
}


// Allocate device buffers and copy inputs/masks
__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_output,   // not used for prolog, provided for symmetry
    const float *host_input,
    const float *host_mask,
    float **device_output_ptr,
    float **device_input_ptr,
    float **device_mask_ptr,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    size_t bytes_in  = static_cast<size_t>(Batch) * Channel * Height * Width * sizeof(float);
    size_t bytes_w   = static_cast<size_t>(Map_out) * Channel * K * K * sizeof(float);
    size_t bytes_out = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);

    // Allocate
    cudaMalloc((void**)device_input_ptr,  bytes_in);
    cudaMalloc((void**)device_mask_ptr,   bytes_w);
    cudaMalloc((void**)device_output_ptr, bytes_out);

    // Copy inputs to device
    cudaMemcpy(*device_input_ptr, host_input, bytes_in, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  bytes_w,  cudaMemcpyHostToDevice);

    // Optional: zero the output
    cudaMemset(*device_output_ptr, 0, bytes_out);
}


// Launch the basic kernel
__host__ void GPUInterface::conv_forward_gpu(
    float *device_output,
    const float *device_input,
    const float *device_mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    const int W_grid = (Width_out  + TILE_WIDTH - 1) / TILE_WIDTH;
    const int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, W_grid * H_grid, Batch);

    conv_forward_kernel<<<gridDim, blockDim>>>(
        device_output, device_input, device_mask,
        Batch, Map_out, Channel, Height, Width, K);

    cudaDeviceSynchronize();
}


// Copy results back and free device allocations
__host__ void GPUInterface::conv_forward_gpu_epilog(
    float *host_output,
    float *device_output,
    float *device_input,
    float *device_mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    size_t bytes_out = static_cast<size_t>(Batch) * Map_out * Height_out * Width_out * sizeof(float);

    // DtoH copy
    cudaMemcpy(host_output, device_output, bytes_out, cudaMemcpyDeviceToHost);

    // Free
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}