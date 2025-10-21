// MP5 Reduction
// Input: A num list of length n
// Output: Sum of the list = list[0] + list[1] + ... + list[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Improved reduction: each thread loads up to 2 elements, sums to shared memory,
// then does a tree reduction with final warp-unrolled steps.
__global__ void total(const float * __restrict__ input,
                      float * __restrict__ output,
                      int len) {
  extern __shared__ float sdata[]; // size = blockDim.x * sizeof(float)

  unsigned int tid  = threadIdx.x;
  unsigned int base = 2u * blockDim.x * blockIdx.x;
  unsigned int i0   = base + tid;
  unsigned int i1   = i0 + blockDim.x;

  // Load up to two elements per thread (guarding OOB with 0 identity)
  float sum = 0.0f;
  if (i0 < (unsigned)len) sum += input[i0];
  if (i1 < (unsigned)len) sum += input[i1];

  sdata[tid] = sum;
  __syncthreads();

  // Reduce in shared memory until 32
  for (unsigned int s = blockDim.x >> 1; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Final warp reduction (no __syncthreads needed within a warp)
  if (tid < 32) {
    volatile float* vsmem = sdata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // Write block's partial sum
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;   // The input 1D list
  float *hostOutput;  // The output list (one partial sum per block)
  //@@ Initialize device input and output pointers
  float *deviceInput  = nullptr;
  float *deviceOutput = nullptr;

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements

  //@@ Allocate GPU memory
  wbCheck(cudaMalloc((void**)&deviceInput,  numInputElements  * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutput, numOutputElements * sizeof(float)));

  //@@ Copy input memory to the GPU
  wbCheck(cudaMemcpy(deviceInput,
                     hostInput,
                     numInputElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(numOutputElements, 1, 1);
  size_t smemBytes = BLOCK_SIZE * sizeof(float);

  //@@ Launch the GPU Kernel and perform CUDA computation
  total<<<dimGrid, dimBlock, smemBytes>>>(deviceInput, deviceOutput, numInputElements);
  wbCheck(cudaGetLastError());
  wbCheck(cudaDeviceSynchronize());

  //@@ Copy the GPU output memory back to the CPU
  wbCheck(cudaMemcpy(hostOutput,
                     deviceOutput,
                     numOutputElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input.
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  //@@ Free the GPU memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
