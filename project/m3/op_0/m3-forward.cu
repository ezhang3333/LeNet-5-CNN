#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#ifndef TILE
#define TILE 16
#endif

#define MAX_CONST_W_ELEMS 16384
__constant__ float Wc[MAX_CONST_W_ELEMS];

__host__ __device__ inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void matmul_conv_fused_const(const float * __restrict__ mask,
                                        const float * __restrict__ input,
                                        float * __restrict__ output,
                                        int Batch, int Map_out, int Channel,
                                        int Height, int Width, int K,
                                        int use_const_w)
{
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const size_t Kc = static_cast<size_t>(Channel) * K * K;
    const size_t Ncols = static_cast<size_t>(Batch) * H_out * W_out;

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;

    for (size_t t = 0; t < (Kc + TILE - 1) / TILE; ++t)
    {
        float a_elt = 0.0f;
        const size_t a_col = t * TILE + threadIdx.x;
        if (row < Map_out && a_col < Kc)
        {
            const int c = static_cast<int>(a_col / (K * K));
            const int rem = static_cast<int>(a_col % (K * K));
            const int p = rem / K;
            const int q = rem % K;

            const size_t mcpq = ((static_cast<size_t>(row) * Channel + c) * K + p) * K + q;

            if (use_const_w && mcpq < MAX_CONST_W_ELEMS)
                a_elt = Wc[mcpq];
            else
                a_elt = mask[mcpq];
        }
        As[threadIdx.y][threadIdx.x] = a_elt;

        float b_elt = 0.0f;
        const size_t b_row = t * TILE + threadIdx.y;
        if (b_row < Kc && col < Ncols)
        {
            const int H_out_local = H_out;
            const int W_out_local = W_out;
            const size_t HW_out = static_cast<size_t>(H_out_local) * W_out_local;

            const int b = static_cast<int>(col / HW_out);
            const size_t hw = static_cast<size_t>(col % HW_out);
            const int h_o = static_cast<int>(hw / W_out_local);
            const int w_o = static_cast<int>(hw % W_out_local);

            const int c = static_cast<int>(b_row / (K * K));
            const int rem = static_cast<int>(b_row % (K * K));
            const int p = rem / K;
            const int q = rem % K;

            const int h_in = h_o + p;
            const int w_in = w_o + q;

            if (b < Batch && c < Channel &&
                h_in >= 0 && h_in < Height &&
                w_in >= 0 && w_in < Width)
            {
                const size_t idx = (((static_cast<size_t>(b) * Channel + c) * Height + h_in) * Width + w_in);
                b_elt = input[idx];
            }
        }
        Bs[threadIdx.y][threadIdx.x] = b_elt;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < Map_out && col < Ncols)
    {
        const int H_out_local = H_out;
        const int W_out_local = W_out;
        const size_t HW_out = static_cast<size_t>(H_out_local) * W_out_local;

        const int b = static_cast<int>(col / HW_out);
        const size_t hw = static_cast<size_t>(col % HW_out);
        const int h_o = static_cast<int>(hw / W_out_local);
        const int w_o = static_cast<int>(hw % W_out_local);

        const size_t out_idx =
            (((static_cast<size_t>(b) * Map_out + row) * H_out_local) + h_o) * W_out_local + w_o;

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
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const size_t in_elems = static_cast<size_t>(Batch) * Channel * Height * Width;
    const size_t out_elems = static_cast<size_t>(Batch) * Map_out * H_out * W_out;
    const size_t m_elems = static_cast<size_t>(Map_out) * Channel * K * K;

    const size_t in_bytes = in_elems * sizeof(float);
    const size_t out_bytes = out_elems * sizeof(float);
    const size_t m_bytes = m_elems * sizeof(float);

    cudaMalloc(reinterpret_cast<void **>(device_input_ptr), in_bytes);
    cudaMalloc(reinterpret_cast<void **>(device_output_ptr), out_bytes);
    cudaMalloc(reinterpret_cast<void **>(device_mask_ptr), m_bytes);

    cudaMemcpy(*device_input_ptr, host_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, m_bytes, cudaMemcpyHostToDevice);
    cudaMemset(*device_output_ptr, 0, out_bytes);

    if (m_elems <= MAX_CONST_W_ELEMS) {
        cudaMemcpyToSymbol(Wc, host_mask, m_bytes, 0, cudaMemcpyHostToDevice);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float * device_output,
                                             const float * device_input,
                                             const float * device_mask,
                                             const int Batch, const int Map_out,
                                             const int Channel, const int Height,
                                             const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const size_t Ncols = static_cast<size_t>(Batch) * H_out * W_out;
    const size_t m_elems = static_cast<size_t>(Map_out) * Channel * K * K;

    dim3 block(TILE, TILE, 1);
    dim3 grid(iDivUp(static_cast<int>(Ncols), TILE), iDivUp(Map_out, TILE), 1);

    int use_const_w = (m_elems <= MAX_CONST_W_ELEMS) ? 1 : 0;

    matmul_conv_fused_const<<<grid, block>>>(device_mask,
                                             device_input,
                                             device_output,
                                             Batch, Map_out, Channel, Height, Width, K,
                                             use_const_w);

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
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    const size_t out_elems = static_cast<size_t>(Batch) * Map_out * H_out * W_out;
    const size_t out_bytes = out_elems * sizeof(float);

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