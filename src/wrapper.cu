/******************************************************************************
 * Copyright (c) 2015 Matthew Nicely
 * Licensed under The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

/******************************************************************************
 * wrapper.cu
 *
 * Used to call CUDA from C++ file.
 *
 ******************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "wrapper.cuh"
#include "opt_kernel.cuh"

__host__ void handleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
      printf( "%s in %s at line %d\n", cudaGetErrorString( err ),	file, line );
      exit( EXIT_FAILURE );
  }
} // HandleError
#define HANDLE_ERROR( err ) (handleError( err, __FILE__, __LINE__ ))

int getGPU_Info(void) {

  int deviceCount = 0;

  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
      printf("There is no device supporting CUDA\n");
  } else if (deviceCount == 1) {
      printf("Found %d CUDA Capable device\n", deviceCount);
  } else if (deviceCount > 1) {
      printf("Found %d CUDA Capable device(s)\n", deviceCount);
      printf("Setting Device to Device 1\n\n");
      cudaSetDevice(1);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;     

  for (dev = 0; dev < deviceCount; ++dev) {

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);

//      printf("Device %d: \"%s\"\n", dev, deviceProp.name);

      // Console log
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
//      printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
//      printf("CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
//      printf("Total amount of global memory: %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
//      printf("Total amount of constant memory: %u bytes\n", (unsigned)deviceProp.totalConstMem);
//      printf("Total amount of shared memory per block: %u bytes\n", (unsigned)deviceProp.sharedMemPerBlock);
//      printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
//      printf("Warp size: %d\n", deviceProp.warpSize);
//      printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
//      printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
//      printf("Maximum sizes of each dimension of a block: %d x %d x %d\n",
//             deviceProp.maxThreadsDim[0],
//             deviceProp.maxThreadsDim[1],
//             deviceProp.maxThreadsDim[2]);
//      printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
//             deviceProp.maxGridSize[0],
//             deviceProp.maxGridSize[1],
//             deviceProp.maxGridSize[2]);
//      printf("\n");
  }
  return (deviceCount);
}

void cuda_function(int num_cities, city_coords *h_coords, best_2opt *gpuResult) {

  int gridSize;       // The actual grid size needed

  // Create variables for GPU
  struct city_coords *d_coords;
  struct best_2opt *d_block;

  // Determine thread size and block size
  gridSize = (num_cities + threadsPerBlock - 1) / threadsPerBlock;

  // Array of structures to hold best result from each block
  struct best_2opt *h_block;
  h_block = new struct best_2opt[gridSize * 3];

  // Allocate memory on GPU
  HANDLE_ERROR(cudaMallocHost((void**)&d_coords, num_cities * sizeof(struct city_coords)));
  HANDLE_ERROR(cudaMallocHost((void**)&d_block, gridSize * sizeof(struct best_2opt)));

  // Copy from CPU to GPU
  HANDLE_ERROR(cudaMemcpy(d_coords, h_coords, num_cities * sizeof(struct city_coords), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_block, h_block, gridSize * sizeof(struct best_2opt), cudaMemcpyHostToDevice));

  //to calculate the number of jobs/2-opt changes and iteration number for each thread
  unsigned long long counter = (long)(num_cities-2)*(long)(num_cities-1)/2;
  unsigned int iterations = (counter/(threadsPerBlock*gridSize)) + 1;

  // Execute kernel
  find_route<<<gridSize, threadsPerBlock>>>(num_cities, d_coords, counter, iterations, d_block);

  // Sync Device
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy from GPU to CPU
  HANDLE_ERROR(cudaMemcpy(h_block, d_block, gridSize * sizeof(struct best_2opt), cudaMemcpyDeviceToHost));

  // Reduction of block results
  for (int i=1; i<gridSize; i++) {
      if (h_block[i].minchange < h_block[0].minchange) h_block[0] = h_block[i];
  }

  // Copy best to structure used by two_opt()
  memcpy((void*)gpuResult, (void*)&h_block[0], sizeof(struct best_2opt));

  // Delete allocate memory
  delete(h_block);
}

void resetGPU() {
  cudaDeviceReset();
}
