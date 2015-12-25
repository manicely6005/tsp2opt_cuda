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

wrapper::wrapper(int num_cities)
{
  gridSize = (num_cities + threadsPerBlock - 1) / threadsPerBlock;

  //to calculate the number of jobs/2-opt changes and iteration number for each thread
  counter = (long)(num_cities-2)*(long)(num_cities-1)/2;
  iterations = (counter/(threadsPerBlock*gridSize)) + 1;

  // Allocate host memory
  h_block = new struct best_2opt[gridSize];

  // Allocate memory on GPU
  HANDLE_ERROR(cudaMalloc((void**)&d_coords, num_cities * sizeof(struct city_coords)));
  HANDLE_ERROR(cudaMalloc((void**)&d_block, gridSize * sizeof(struct best_2opt)));
}

wrapper::~wrapper()
{
  delete(h_block);
  // Replaces multiple cudaFree's
  HANDLE_ERROR(cudaDeviceReset());
}

void wrapper::getGPU_Info(void) {

  int deviceCount = 0;

  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
      printf("There is no device supporting CUDA\n");
  } else if (deviceCount == 1) {
      printf("Found %d CUDA Capable device\n", deviceCount);
  } else if (deviceCount > 1) {
      printf("Found %d CUDA Capable device(s)\n", deviceCount);
      printf("Setting Device to Device 1\n\n");
      HANDLE_ERROR(cudaSetDevice(0));
  }
}

void wrapper::cuda_function(int num_cities, city_coords *h_coords, best_2opt *gpuResult) {

  HANDLE_ERROR(cudaMemcpy(d_coords, h_coords, num_cities * sizeof(struct city_coords), cudaMemcpyHostToDevice));

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
}
