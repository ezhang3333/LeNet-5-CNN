#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256

// Tunables for unroll kernel launch
#ifndef UNROLL_TILE_X
#define UNROLL_TILE_X 32
#endif
#ifndef UNROLL_TILE_Y
#define UNROLL_TILE_Y 8
#endif

// Build im2col matrix: (Channel*K*K) x (Batch*H_out*W_out)
//   row = c*K*K + p*K + q
//   col = b*(H_out*W_out) + h_out*W_out + w_out
// value = input[b, c, h_out+p, w_out+q]
__global__ void matrix_unrolling_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    const size_t rows_unrolled = static_cast<size_t>(Channel) * K * K;            
    const size_t cols_unrolled = static_cast<size_t>(Batch) * H_out * W_out;      

    const size_t row = static_cast<size_t>(blockIdx.y) * UNROLL_TILE_Y + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * UNROLL_TILE_X + threadIdx.x;

    if (row >= rows_unrolled || col >= cols_unrolled) return;

    const int KK  = K * K;
    const int c   = static_cast<int>(row / KK);
    const int r2  = static_cast<int>(row % KK);
    const int p   = r2 / K;
    const int q   = r2 % K;

    const size_t HW_out = static_cast<size_t>(H_out) * W_out;
    const int b     = static_cast<int>(col / HW_out);
    const size_t hw = static_cast<size_t>(col % HW_out);
    const int h_out = static_cast<int>(hw / W_out);
    const int w_out = static_cast<int>(hw % W_out);

    const int h_in = h_out + p;
    const int w_in = w_out + q;

    float val = 0.0f;
    if (b < Batch && c < Channel &&
        h_in >= 0 && h_in < Height &&
        w_in >= 0 && w_in < Width) {
        const size_t idx =
            (((static_cast<size_t>(b) * Channel + c) * Height + h_in) * Width + w_in);
        val = input[idx];
    }

    output[row * cols_unrolled + col] = val;
}

// Permutes the matmul result (do not modify)
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *,
                                                    const float *host_input,
                                                    const float *host_mask,
                                                    float **device_output_ptr,
                                                    float **device_input_ptr,
                                                    float **device_mask_ptr,
                                                    const int Batch, const int Map_out,
                                                    const int Channel, const int Height,
                                                    const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    const size_t in_elems  = static_cast<size_t>(Batch) * Channel * Height * Width;
    const size_t out_elems = static_cast<size_t>(Batch) * Map_out * H_out * W_out;
    const size_t m_elems   = static_cast<size_t>(Map_out) * Channel * K * K;

    const size_t in_bytes  = in_elems  * sizeof(float);
    const size_t out_bytes = out_elems * sizeof(float);
    const size_t m_bytes   = m_elems   * sizeof(float);

    // Allocate device buffers
    cudaMalloc(reinterpret_cast<void **>(device_input_ptr),  in_bytes);
    cudaMalloc(reinterpret_cast<void **>(device_output_ptr), out_bytes);
    cudaMalloc(reinterpret_cast<void **>(device_mask_ptr),   m_bytes);

    // Copy inputs to device
    cudaMemcpy(*device_input_ptr, host_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,  host_mask,  m_bytes,  cudaMemcpyHostToDevice);

    // Clear output
    cudaMemset(*device_output_ptr, 0, out_bytes);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output,
                                             const float *device_input,
                                             const float *device_mask,
                                             const int Batch, const int Map_out,
                                             const int Channel, const int Height,
                                             const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;

    const size_t M = static_cast<size_t>(Channel) * K * K;          
    const size_t N = static_cast<size_t>(Batch) * H_out * W_out;   
    const size_t unroll_bytes = M * N * sizeof(float);

    float *unrolled_matrix = nullptr;  
    float *matmul_output   = nullptr;   
    cudaMalloc(reinterpret_cast<void**>(&unrolled_matrix), unroll_bytes);
    cudaMalloc(reinterpret_cast<void**>(&matmul_output),
               static_cast<size_t>(Map_out) * N * sizeof(float));

    dim3 block(UNROLL_TILE_X, UNROLL_TILE_Y, 1);
    dim3 grid(
        static_cast<unsigned int>((N + UNROLL_TILE_X - 1) / UNROLL_TILE_X),
        static_cast<unsigned int>((M + UNROLL_TILE_Y - 1) / UNROLL_TILE_Y),
        1
    );
    matrix_unrolling_kernel<<<grid, block>>>(device_input, unrolled_matrix,
                                             Batch, Channel, Height, Width, K);

    dim3 matmul_grid_dim(
        static_cast<unsigned int>((N - 1) / MATMUL_TILE_WIDTH + 1),
        static_cast<unsigned int>((Map_out - 1) / MATMUL_TILE_WIDTH + 1),
        1
    );
    dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim>>>(
        device_mask,      
        unrolled_matrix,    
        matmul_output,        
        Map_out,               
        static_cast<int>(M),   
        static_cast<int>(M),   
        static_cast<int>(N),  
        Map_out, 
        static_cast<int>(N)
    );

    const int out_image_size = H_out * W_out;
    dim3 permute_grid((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_grid, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output,
                                                    float *device_output,
                                                    float *device_input,
                                                    float *device_mask,
                                                    const int Batch, const int Map_out,
                                                    const int Channel, const int Height,
                                                    const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width  - K + 1;
    const size_t out_elems = static_cast<size_t>(Batch) * Map_out * H_out * W_out;
    const size_t out_bytes = out_elems * sizeof(float);

    // Copy back
    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    // Free device inputs/outputs
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
