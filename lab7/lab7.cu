// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this (threads per block). Each block scans 2*BLOCK_SIZE elements.

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// -----------------------------------------------------------------------------
//@@ Work-efficient per-block inclusive scan (Brent–Kung) on up to 2*blockDim.x elements.
//   - Loads two elements per thread into shared memory (padding with 0)
//   - Performs reduction (upsweep) then post-scan (downsweep) to produce INCLUSIVE scan
//   - Writes scanned segment to 'out'
//   - Writes per-block total (last valid element) to 'blockSums' if non-null
//   Follows the lecture’s pattern (reduction tree + distribution tree). :contentReference[oaicite:1]{index=1}
__global__ void scanBlockKernel(const float *__restrict__ in,
                                float *__restrict__ out,
                                float *__restrict__ blockSums,
                                int n) {
  __shared__ float T[2 * BLOCK_SIZE];

  const int t = threadIdx.x;
  const int start = 2 * blockDim.x * blockIdx.x; // base index for this block's segment
  const int ai = start + t;
  const int bi = start + t + blockDim.x;

  // Load with boundary padding (identity = 0 for sum)
  T[t]                 = (ai < n) ? in[ai] : 0.0f;
  T[t + blockDim.x]    = (bi < n) ? in[bi] : 0.0f;

  // ---------------- Upsweep (reduction) ----------------
  // section size is 2 * blockDim.x, but pad beyond 'n' with 0s
  int stride = 1;
  int section = 2 * blockDim.x;
  while (stride < section) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index < section) {
      T[index] += T[index - stride];
    }
    stride <<= 1;
  }

  // ---------------- Downsweep (post-scan for inclusive) ----------------
  stride = blockDim.x / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < section) {
      T[index + stride] += T[index];
    }
    stride >>= 1;
  }
  __syncthreads();

  // Write scanned values back to global memory (inclusive scan of this segment)
  if (ai < n) out[ai] = T[t];
  if (bi < n) out[bi] = T[t + blockDim.x];

  // Write this block's total (last valid element in this segment) to blockSums
  if (blockSums != nullptr && t == 0) {
    int validCount = min(section, n - start);
    float total = 0.0f;
    if (validCount > 0) {
      total = T[validCount - 1];
    }
    blockSums[blockIdx.x] = total;
  }
}

// -----------------------------------------------------------------------------
//@@ Single-block in-place inclusive scan for the block-sums array.
//   Launch with <<<1, threadsForSums>>> where 2*threadsForSums >= m.
__global__ void scanBlockSumsSingleBlock(float *__restrict__ data, int m) {
  __shared__ float T[2 * BLOCK_SIZE];

  const int t = threadIdx.x;
  const int section = 2 * blockDim.x;

  T[t]                 = (t < m) ? data[t] : 0.0f;
  T[t + blockDim.x]    = ((t + blockDim.x) < m) ? data[t + blockDim.x] : 0.0f;

  int stride = 1;
  while (stride < section) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index < section) {
      T[index] += T[index - stride];
    }
    stride <<= 1;
  }

  stride = blockDim.x / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < section) {
      T[index + stride] += T[index];
    }
    stride >>= 1;
  }
  __syncthreads();

  if (t < m) data[t] = T[t];
  if ((t + blockDim.x) < m) data[t + blockDim.x] = T[t + blockDim.x];
}

// -----------------------------------------------------------------------------
//@@ Add scanned block sums to each block's scanned segment to complete the global scan
__global__ void addScannedBlockSums(float *__restrict__ out,
                                    const float *__restrict__ scannedBlockSums,
                                    int n) {
  const int t = threadIdx.x;
  const int section = 2 * blockDim.x;
  const int start = section * blockIdx.x;
  const int ai = start + t;
  const int bi = start + t + blockDim.x;

  float addVal = (blockIdx.x == 0) ? 0.0f : scannedBlockSums[blockIdx.x - 1];

  if (ai < n) out[ai] += addVal;
  if (bi < n) out[bi] += addVal;
}

// -----------------------------------------------------------------------------
// (Not used; kept to match the provided skeleton signature.)
//@@ You may ignore this kernel; we launch the specific kernels from host in main().
__global__ void scan(float *input, float *output, int len) {
  // Intentionally left empty; see main() for the multi-kernel flow.
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceBlockSums = nullptr; //@@ auxiliary array for per-block sums
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));

  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput,  numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  const int threadsPerBlock = BLOCK_SIZE;
  const int elemsPerBlock   = 2 * threadsPerBlock;
  const int numBlocks       = (numElements + elemsPerBlock - 1) / elemsPerBlock;

  if (numBlocks > 1) {
    wbCheck(cudaMalloc((void **)&deviceBlockSums, numBlocks * sizeof(float)));
  }

  //@@ Modify this to complete the functionality of the scan
  //@@ on the device

  scanBlockKernel<<<numBlocks, threadsPerBlock>>>(deviceInput,
                                                  deviceOutput,
                                                  deviceBlockSums,
                                                  numElements);
  wbCheck(cudaGetLastError());
  wbCheck(cudaDeviceSynchronize());

  if (numBlocks > 1) {
    int threadsForSums = threadsPerBlock;
    while (2 * threadsForSums >= numBlocks && threadsForSums > 1) {
      break;
    }

    scanBlockSumsSingleBlock<<<1, threadsPerBlock>>>(deviceBlockSums, numBlocks);
    wbCheck(cudaGetLastError());
    wbCheck(cudaDeviceSynchronize());

    addScannedBlockSums<<<numBlocks, threadsPerBlock>>>(deviceOutput,
                                                        deviceBlockSums,
                                                        numElements);
    wbCheck(cudaGetLastError());
    wbCheck(cudaDeviceSynchronize());
  }

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  //@@  Free GPU Memory
  if (deviceBlockSums) cudaFree(deviceBlockSums);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
