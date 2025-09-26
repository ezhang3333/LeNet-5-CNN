#include <wb.h>
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define K_WIDTH 3
#define RADIUS 1
#define TILE 8                              // output tile width per dim
#define BLOCK_EDGE (TILE + 2*RADIUS)        // input tile width (with halo)

//@@ Define constant memory for device kernel here
__constant__ float d_K[K_WIDTH*K_WIDTH*K_WIDTH];

__global__ void conv3d(float *input, float *output,
                       const int z_size, const int y_size, const int x_size) {
  //@@ Insert kernel code here (3D Strategy-2 tiling with constant memory)
  __shared__ float tile[BLOCK_EDGE][BLOCK_EDGE][BLOCK_EDGE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Global coords this thread loads into shared memory (input-tile coords)
  int gx_in = blockIdx.x * TILE + tx - RADIUS;
  int gy_in = blockIdx.y * TILE + ty - RADIUS;
  int gz_in = blockIdx.z * TILE + tz - RADIUS;

  // Load with zero-padding at boundaries
  float v = 0.0f;
  if (gz_in >= 0 && gz_in < z_size &&
      gy_in >= 0 && gy_in < y_size &&
      gx_in >= 0 && gx_in < x_size) {
    v = input[(gz_in * y_size + gy_in) * x_size + gx_in];
  }
  tile[tz][ty][tx] = v;

  __syncthreads();

  // Only inner TILE^3 threads compute an output
  if (tx >= RADIUS && tx < RADIUS + TILE &&
      ty >= RADIUS && ty < RADIUS + TILE &&
      tz >= RADIUS && tz < RADIUS + TILE) {

    int gx_out = blockIdx.x * TILE + (tx - RADIUS);
    int gy_out = blockIdx.y * TILE + (ty - RADIUS);
    int gz_out = blockIdx.z * TILE + (tz - RADIUS);

    if (gx_out < x_size && gy_out < y_size && gz_out < z_size) {
      float acc = 0.0f;
      #pragma unroll
      for (int kz = 0; kz < K_WIDTH; ++kz) {
        #pragma unroll
        for (int ky = 0; ky < K_WIDTH; ++ky) {
          #pragma unroll
          for (int kx = 0; kx < K_WIDTH; ++kx) {
            float w = d_K[(kz*K_WIDTH + ky)*K_WIDTH + kx];
            float x = tile[tz + kz - RADIUS][ty + ky - RADIUS][tx + kx - RADIUS];
            acc += w * x;
          }
        }
      }
      output[(gz_out * y_size + gy_out) * x_size + gx_out] = acc;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;

  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput = nullptr;
  float *deviceOutput = nullptr;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput  = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = (int)hostInput[0];
  y_size = (int)hostInput[1];
  x_size = (int)hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  const size_t nElem = (size_t)z_size * y_size * x_size;
  const size_t bytes = nElem * sizeof(float);

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  wbCheck(cudaMalloc((void **)&deviceInput,  bytes));
  wbCheck(cudaMalloc((void **)&deviceOutput, bytes));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput,  hostInput + 3, bytes, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(d_K, hostKernel, 27 * sizeof(float), 0, cudaMemcpyHostToDevice));

  //@@ Initialize grid and block dimensions here
  dim3 block(BLOCK_EDGE, BLOCK_EDGE, BLOCK_EDGE); // (TILE+2)^3 threads
  dim3 grid( (x_size + TILE - 1) / TILE,
             (y_size + TILE - 1) / TILE,
             (z_size + TILE - 1) / TILE );

  //@@ Launch the GPU kernel here
  conv3d<<<grid, block>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  wbCheck(cudaGetLastError());
  wbCheck(cudaDeviceSynchronize());

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(hostOutput + 3, deviceOutput, bytes, cudaMemcpyDeviceToHost));

  // Set the output dimensions for correctness checking
  hostOutput[0] = (float)z_size;
  hostOutput[1] = (float)y_size;
  hostOutput[2] = (float)x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));

  // Free host memory
  free(hostInput);
  free(hostKernel);
  free(hostOutput);
  return 0;
}
