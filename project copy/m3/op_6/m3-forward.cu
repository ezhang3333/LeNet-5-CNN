#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#ifndef JT_T
#define JT_T 64
#endif
#ifndef JT_U
#define JT_U 16
#endif
#ifndef JT_S
#define JT_S (JT_T / JT_U)
#endif

#if (JT_T != JT_U * JT_S)
#error "Joint tiling requires T == U * S"
#endif

__host__ __device__ inline int iDivUp(int a, int b) { 
    return (a + b - 1) / b; 
}

__global__ void matmul_conv_joint(const float* __restrict__ mask,
                                  const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int Batch, int Map_out, int Channel,
                                  int Height, int Width, int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const size_t Kc = static_cast<size_t>(Channel) * K * K;
    const size_t Ncols = static_cast<size_t>(Batch) * H_out * W_out;
    const size_t HWout = static_cast<size_t>(H_out) * W_out;

    const int row0 = blockIdx.y * JT_T;
    const int col0 = blockIdx.x * JT_U;

    const int tRow = threadIdx.x;
    const int gRow = row0 + tRow;

    const int sLoad = tRow / JT_U;
    const int uLoad = tRow % JT_U;

    __shared__ float Ns[JT_S][JT_U];

    float acc[JT_U];
    #pragma unroll
    for (int u = 0; u < JT_U; ++u) acc[u] = 0.0f;

    const int iters = static_cast<int>((Kc + JT_S - 1) / JT_S);
    for (int it = 0; it < iters; ++it) {
        const size_t kbase = static_cast<size_t>(it) * JT_S;

        float a_reg[JT_S];
        #pragma unroll
        for (int s = 0; s < JT_S; ++s) {
            const size_t kIdx = kbase + s;
            float a = 0.f;
            if (gRow < Map_out && kIdx < Kc) {
                const int c = static_cast<int>(kIdx / (K * K));
                const int rem = static_cast<int>(kIdx % (K * K));
                const int p = rem / K;
                const int q = rem % K;

                const size_t mcpq = ((static_cast<size_t>(gRow) * Channel + c) * K + p) * K + q;
                a = mask[mcpq];
            }
            a_reg[s] = a;
        }

        const size_t nRow = kbase + static_cast<size_t>(sLoad);
        const size_t nCol = col0 + static_cast<size_t>(uLoad);
        float n_val = 0.f;
        if (nRow < Kc && nCol < Ncols) {
            const int b = static_cast<int>(nCol / HWout);
            const size_t hw = static_cast<size_t>(nCol % HWout);
            const int h_o = static_cast<int>(hw / W_out);
            const int w_o = static_cast<int>(hw % W_out);

            const int c = static_cast<int>(nRow / (K * K));
            const int rem = static_cast<int>(nRow % (K * K));
            const int p = rem / K;
            const int q = rem % K;

            const int h_in = h_o + p;
            const int w_in = w_o + q;

            if (b < Batch && c < Channel &&
                h_in >= 0 && h_in < Height &&
                w_in >= 0 && w_in < Width) {
                const size_t inIdx =
                    (((static_cast<size_t>(b) * Channel + c) * Height + h_in) * Width + w_in);
                n_val = input[inIdx];
            }
        }
        Ns[sLoad][uLoad] = n_val;

        __syncthreads();

        #pragma unroll
        for (int s = 0; s < JT_S; ++s) {
            const float a = a_reg[s];
            #pragma unroll
            for (int u = 0; u < JT_U; ++u) {
                acc[u] += a * Ns[s][u];
            }
        }

        __syncthreads();
    }

    if (gRow < Map_out) {
        #pragma unroll
        for (int u = 0; u < JT_U; ++u) {
            const size_t gCol = col0 + static_cast<size_t>(u);
            if (gCol < Ncols) {
                const int b = static_cast<int>(gCol / HWout);
                const size_t hw = static_cast<size_t>(gCol % HWout);
                const int h_o = static_cast<int>(hw / W_out);
                const int w_o = static_cast<int>(hw % W_out);

                const size_t outIdx =
                    (((static_cast<size_t>(b) * Map_out + gRow) * H_out) + h_o) * W_out + w_o;
                output[outIdx] = acc[u];
            }
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float* /*host_output*/,
                                                    const float* host_input,
                                                    const float* host_mask,
                                                    float** device_output_ptr,
                                                    float** device_input_ptr,
                                                    float** device_mask_ptr,
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
}

__host__ void GPUInterface::conv_forward_gpu(float* device_output,
                                             const float* device_input,
                                             const float* device_mask,
                                             const int Batch, const int Map_out,
                                             const int Channel, const int Height,
                                             const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const size_t Ncols = static_cast<size_t>(Batch) * H_out * W_out;

    dim3 block(JT_T, 1, 1);
    dim3 grid(iDivUp(static_cast<int>(Ncols), JT_U),
              iDivUp(Map_out, JT_T),
              1);

    matmul_conv_joint<<<grid, block>>>(device_mask,
                                       device_input,
                                       device_output,
                                       Batch, Map_out, Channel, Height, Width, K);

    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float* host_output,
                                                    float* device_output,
                                                    float* device_input,
                                                    float* device_mask,
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