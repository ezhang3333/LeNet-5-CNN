#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;

  float acc = 0.0f;

  // Number of tiles to cover the inner dimension (numAColumns == numBRows)
  int phases = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int ph = 0; ph < phases; ++ph) {
    int aCol = ph * TILE_WIDTH + tx;
    int bRow = ph * TILE_WIDTH + ty;

    // Load a tile of A into shared memory (guarded)
    if (row < numARows && aCol < numAColumns) {
      As[ty][tx] = A[row * numAColumns + aCol];
    } else {
      As[ty][tx] = 0.0f;
    }

    // Load a tile of B into shared memory (guarded)
    if (bRow < numBRows && col < numBColumns) {
      Bs[ty][tx] = B[bRow * numBColumns + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Partial dot-product for this tile
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  // Write result (guarded)
  if (row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = acc;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);

  //@@ Allocate GPU memory here
  float *devA, *devB, *devC;
  wbCheck(cudaMalloc((void **)&devA, sizeof(float) * numARows * numAColumns));
  wbCheck(cudaMalloc((void **)&devB, sizeof(float) * numBRows * numBColumns));
  wbCheck(cudaMalloc((void **)&devC, sizeof(float) * numCRows * numCColumns));

  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(devA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(devB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
               (numCRows   + TILE_WIDTH - 1) / TILE_WIDTH,
               1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<gridDim, blockDim>>>(devA, devB, devC,
                                              numARows, numAColumns,
                                              numBRows, numBColumns,
                                              numCRows, numCColumns);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, devC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));
  

  //@@ Free the GPU memory here
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
