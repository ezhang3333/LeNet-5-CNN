#include <cmath>
#include <iostream>
#include <mma.h>
#include "gpu-new-forward.h"

using namespace nvcuda;

#ifndef TILE
#define TILE 16
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

__host__ __device__ inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void matmul_conv_fused_tensorcore(const float* __restrict__ mask,
                                             const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int Batch, int Map_out, int Channel,
                                             int Height, int Width, int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    const int M = Map_out;
    const int Kc = Channel * K * K;
    const int N = Batch * H_out * W_out;

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;

    const int row_base = tile_m * WMMA_M;
    const int col_base = tile_n * WMMA_N;

    const int laneId = threadIdx.x % 32;
    if (threadIdx.x >= 32) return;

    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    extern __shared__ float shmem[];
    float* A_tile = shmem;
    float* B_tile = A_tile + WMMA_M * WMMA_K;
    float* C_tile = B_tile + WMMA_K * WMMA_N;

    using TF32_TYPE = wmma::precision::tf32;

    wmma::fragment<wmma::matrix_a,
                   WMMA_M, WMMA_N, WMMA_K,
                   TF32_TYPE,
                   wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b,
                   WMMA_M, WMMA_N, WMMA_K,
                   TF32_TYPE,
                   wmma::col_major> b_frag;

    for (int k0 = 0; k0 < Kc; k0 += WMMA_K) {
        const int numA = WMMA_M * WMMA_K;
        for (int idx = laneId; idx < numA; idx += 32) {
            int i = idx / WMMA_K;
            int j = idx % WMMA_K;

            int m = row_base + i;
            int kk = k0 + j;

            float val = 0.0f;
            if (m < M && kk < Kc) {
                int KK = K * K;
                int c = kk / KK;
                int rem = kk % KK;
                int p = rem / K;
                int q = rem % K;

                size_t mcpq = (((size_t)m * Channel + c) * K + p) * K + q;
                val = mask[mcpq];
            }
            A_tile[i * WMMA_K + j] = val;
        }

        const int numB = WMMA_K * WMMA_N;
        for (int idx = laneId; idx < numB; idx += 32) {
            int i = idx / WMMA_N;
            int j = idx % WMMA_N;

            int kk = k0 + i;
            int n = col_base + j;

            float val = 0.0f;
            if (kk < Kc && n < N) {
                int HW_out = H_out * W_out;
                int b = n / HW_out;
                int hw = n % HW_out;
                int h_o = hw / W_out;
                int w_o = hw % W_out;

                int KK = K * K;
                int c = kk / KK;
                int rem = kk % KK;
                int p = rem / K;
                int q = rem % K;

                int h_in = h_o + p;
                int w_in = w_o + q;

                if (b < Batch && c < Channel &&
                    h_in >= 0 && h_in < Height &&
                    w_in >= 0 && w_in < Width) {
                    size_t idx_input =
                        (((size_t)b * Channel + c) * Height + h_in) * Width + w_in;
                    val = input[idx_input];
                }
            }
            B_tile[j * WMMA_K + i] = val;
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, A_tile, WMMA_K);
        wmma::load_matrix_sync(b_frag, B_tile, WMMA_K);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(C_tile, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    const int numC = WMMA_M * WMMA_N;
    for (int idx = laneId; idx < numC; idx += 32) {
        int i = idx / WMMA_N;
        int j = idx % WMMA_N;

        int m = row_base + i;
        int n = col_base + j;

        if (m < M && n < N) {
            float val = C_tile[i * WMMA_N + j];

            int HW_out = H_out * W_out;
            int b = n / HW_out;
            int hw = n % HW_out;
            int h_o = hw / W_out;
            int w_o = hw % W_out;

            size_t out_idx =
                (((size_t)b * Map_out + m) * H_out + h_o) * W_out + w_o;

            output[out_idx] = val;
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

    cudaMalloc(reinterpret_cast<void**>(device_input_ptr), in_bytes);
    cudaMalloc(reinterpret_cast<void**>(device_output_ptr), out_bytes);
    cudaMalloc(reinterpret_cast<void**>(device_mask_ptr), m_bytes);

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

    const int M = Map_out;
    const int Kc = Channel * K * K;
    const int N = Batch * H_out * W_out;

    dim3 block(32, 1, 1);
    dim3 grid((N + WMMA_N - 1) / WMMA_N,
              (M + WMMA_M - 1) / WMMA_M,
              1);

    size_t shmemBytes =
        (WMMA_M * WMMA_K + WMMA_K * WMMA_N + WMMA_M * WMMA_N) * sizeof(float);

    matmul_conv_fused_tensorcore<<<grid, block, shmemBytes>>>(
        device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);

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
        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "."
                  << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, "
                  << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, "
                  << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
