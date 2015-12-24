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

void getGPU_Info(void) {

  int deviceCount = 0;

  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
      printf("There is no device supporting CUDA\n");
  } else if (deviceCount == 1) {
      printf("Found %d CUDA Capable device\n", deviceCount);
  } else if (deviceCount > 1) {
      printf("Found %d CUDA Capable device(s)\n", deviceCount);
      printf("Setting Device to Device 1\n\n");
      cudaSetDevice(0);
  }
}

void cuda_function(int num_cities, city_coords *h_coords, best_2opt *gpuResult) {

  int gridSize;       // The actual grid size needed

  // Create variables for CPU
  struct best_2opt *h_block;

  // Create variables for GPU
  struct city_coords *d_coords;
  struct best_2opt *d_block;

  // Determine thread size and block size
  gridSize = (num_cities + threadsPerBlock - 1) / threadsPerBlock;

  // If Jetson TK1 ran Toolkit 7.x
  // Since the Jetson uses integrated memory, the proper way would be to use mapped memory.
  // Since the memory for orderCoords is allocated in algorithms.cpp, the correct way
  // to pin to memory would be to use cudaHostRegister() and then cudaFreeHostRegister to
  // free from memory. Unfortunately, cudaHostRegister doesn't work with the combination
  // Linux, Arm7l, and CUDA toolkit 6.5
  //  HANDLE_ERROR(cudaHostRegister(h_coords, num_cities * sizeof(struct city_coords), cudaHostRegisterMapped));

#ifdef ARM
  // Allocate pinned memory
  HANDLE_ERROR(cudaHostAlloc((void**)&h_block, gridSize * sizeof(struct best_2opt), cudaHostAllocMap$

  // Get pointer for pinned memory
  HANDLE_ERROR(cudaHostGetDevicePointer((void**)&d_block, (void*)h_block, 0));

  // Allocate memory on GPU
  HANDLE_ERROR(cudaMalloc((void**)&d_coords, num_cities * sizeof(struct city_coords)));

  // Copy from CPU to GPU
  HANDLE_ERROR(cudaMemcpy(d_coords, h_coords, num_cities * sizeof(struct city_coords), cudaMemcpyHostToDevice));
#else
  // Allocate host memory
  h_block = new struct best_2opt[gridSize];

  // Allocate memory on GPU
  HANDLE_ERROR(cudaMalloc((void**)&d_coords, num_cities * sizeof(struct city_coords)));
  HANDLE_ERROR(cudaMalloc((void**)&d_block, gridSize * sizeof(struct best_2opt)));

  // Copy from CPU to GPU
  HANDLE_ERROR(cudaMemcpy(d_coords, h_coords, num_cities * sizeof(struct city_coords), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_block, h_block, gridSize * sizeof(struct best_2opt), cudaMemcpyHostToDevice));
#endif

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
  cudaFree(d_coords);
  cudaFree(d_block);
#ifdef ARM
  cudaFreeHost(h_block);
#else
  delete(h_block);
#endif
}

void initGPU() {
  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
}

void resetGPU() {
  cudaDeviceReset();
}
