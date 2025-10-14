#include <wb.h>

#define HISTOGRAM_LENGTH 256

// Error checking macro
#define wbCheck(stmt) do { \
  cudaError_t err = stmt; \
  if (err != cudaSuccess) { \
    wbLog(ERROR, "Failed to run stmt ", #stmt); \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
    return -1; \
  } \
} while(0)

//@@ GPU Kernels

__global__ void floatToUchar(float *input, unsigned char *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = (unsigned char)(255.0f * input[i]);
    }
}

__global__ void rgbToGray(unsigned char *input, unsigned char *gray, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        int rgbIdx = i * 3;
        unsigned char r = input[rgbIdx];
        unsigned char g = input[rgbIdx + 1];
        unsigned char b = input[rgbIdx + 2];
        gray[i] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

__global__ void computeHistogram(unsigned char *gray, int width, int height, unsigned int *hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        atomicAdd(&(hist[gray[i]]), 1);
    }
}

__global__ void computeCDF(float *cdf, unsigned int *hist, int size, int totalPixels) {
    __shared__ float temp[HISTOGRAM_LENGTH];
    int t = threadIdx.x;
    if (t < HISTOGRAM_LENGTH) {
        temp[t] = (float)hist[t] / totalPixels;
    }
    __syncthreads();

    // inclusive scan (sequential, fine for 256 elements)
    for (int stride = 1; stride < HISTOGRAM_LENGTH; stride *= 2) {
        float val = 0.0f;
        if (t >= stride) {
            val = temp[t - stride];
        }
        __syncthreads();
        temp[t] += val;
        __syncthreads();
    }

    if (t < HISTOGRAM_LENGTH)
        cdf[t] = temp[t];
}

__global__ void applyEqualization(unsigned char *input, unsigned char *output,
                                  float *cdf, float cdfMin, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        unsigned char val = input[i];
        float equalized = 255.0f * (cdf[val] - cdfMin) / (1.0f - cdfMin);
        if (equalized < 0.0f) equalized = 0.0f;
        if (equalized > 255.0f) equalized = 255.0f;
        output[i] = (unsigned char)(equalized);
    }
}

__global__ void ucharToFloat(unsigned char *input, float *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = ((float)input[i]) / 255.0f;
    }
}

// Main function
int main(int argc, char **argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    const char *inputImageFile;

    // 1. Parse args
    args = wbArg_read(argc, argv);
    inputImageFile = wbArg_getInputFile(args, 0);

    // 2. Import image
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    int imgSize = imageWidth * imageHeight * imageChannels;
    int graySize = imageWidth * imageHeight;

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    // 3. Allocate device memory
    float *deviceInputImageData, *deviceOutputImageData;
    unsigned char *deviceUcharImage, *deviceGrayImage, *deviceEqualizedImage;
    unsigned int *deviceHistogram;
    float *deviceCDF;

    wbCheck(cudaMalloc(&deviceInputImageData, imgSize * sizeof(float)));
    wbCheck(cudaMalloc(&deviceUcharImage, imgSize * sizeof(unsigned char)));
    wbCheck(cudaMalloc(&deviceGrayImage, graySize * sizeof(unsigned char)));
    wbCheck(cudaMalloc(&deviceEqualizedImage, imgSize * sizeof(unsigned char)));
    wbCheck(cudaMalloc(&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
    wbCheck(cudaMalloc(&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
    wbCheck(cudaMalloc(&deviceOutputImageData, imgSize * sizeof(float)));

    wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData,
                       imgSize * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));

    int blockSize = 256;
    int gridSize = (imgSize + blockSize - 1) / blockSize;
    int grayGrid = (graySize + blockSize - 1) / blockSize;

    // 4. Run kernels step by step
    floatToUchar<<<gridSize, blockSize>>>(deviceInputImageData, deviceUcharImage, imgSize);
    cudaDeviceSynchronize();

    rgbToGray<<<grayGrid, blockSize>>>(deviceUcharImage, deviceGrayImage, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    computeHistogram<<<grayGrid, blockSize>>>(deviceGrayImage, imageWidth, imageHeight, deviceHistogram);
    cudaDeviceSynchronize();

    computeCDF<<<1, HISTOGRAM_LENGTH>>>(deviceCDF, deviceHistogram, HISTOGRAM_LENGTH, graySize);
    cudaDeviceSynchronize();

    // 5. Copy CDF to host to find cdfMin
    float hostCDF[HISTOGRAM_LENGTH];
    wbCheck(cudaMemcpy(hostCDF, deviceCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
    float cdfMin = 1.0f;
    for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
        if (hostCDF[i] > 0.0f) {
            cdfMin = hostCDF[i];
            break;
        }
    }

    // 6. Apply equalization to all channels
    applyEqualization<<<gridSize, blockSize>>>(deviceUcharImage, deviceEqualizedImage,
                                               deviceCDF, cdfMin, imgSize);
    cudaDeviceSynchronize();

    // 7. Convert back to float
    ucharToFloat<<<gridSize, blockSize>>>(deviceEqualizedImage, deviceOutputImageData, imgSize);
    cudaDeviceSynchronize();

    // 8. Copy to host
    wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                       imgSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 9. Output result
    wbSolution(args, outputImage);

    // 10. Cleanup
    cudaFree(deviceInputImageData);
    cudaFree(deviceUcharImage);
    cudaFree(deviceGrayImage);
    cudaFree(deviceEqualizedImage);
    cudaFree(deviceHistogram);
    cudaFree(deviceCDF);
    cudaFree(deviceOutputImageData);

    return 0;
}
