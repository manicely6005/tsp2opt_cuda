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
  } else {
    printf("Found %d CUDA Capable device(s)\n", deviceCount);
  }
  
  int dev, driverVersion = 0, runtimeVersion = 0;     
  
  for (dev = 0; dev < deviceCount; ++dev) {
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    
    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total amount of global memory: %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
    printf("Total amount of constant memory: %u bytes\n", (unsigned)deviceProp.totalConstMem); 
    printf("Total amount of shared memory per block: %u bytes\n", (unsigned)deviceProp.sharedMemPerBlock);
    printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum sizes of each dimension of a block: %d x %d x %d\n",
	   deviceProp.maxThreadsDim[0],
	   deviceProp.maxThreadsDim[1],
	   deviceProp.maxThreadsDim[2]);
    printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
	   deviceProp.maxGridSize[0],
	   deviceProp.maxGridSize[1],
	   deviceProp.maxGridSize[2]);
    printf("\n");
  }
  return (deviceCount);
}

void cuda_function(int *route, int distance, int num_cities, city_coords *coords) {

  int blockSize;      // The launch  returned block size
  int gridSize;       // The actual grid size needed
  
  struct best_2opt zero;
  struct best_2opt out;

  printf("num_cities = %d\n", num_cities);

  for (int i=0; i<num_cities; i++) {
	   printf("coords[i].x = %f ", coords[i].x);
      printf("coords[i].y = %f\n", coords[i].y);
  }
  printf("\n");

  zero.i = 1;
  zero.j = 2;
  zero.minchange = 500;

  // Create variables for GPU
//  int *d_route;
  city_coords *d_coords;

  // Allocate memory on GPU
//  HANDLE_ERROR(cudaMalloc((void**)&d_route, (num_cities+1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&d_coords, num_cities * sizeof(d_coords)));
  
  // Copy from CPU to GPU
//  HANDLE_ERROR(cudaMemcpy(d_route, route, (num_cities+1) * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_coords, coords, num_cities * sizeof(d_coords), cudaMemcpyHostToDevice));

  /* This is a wrapper function which allows the wrapper file to copy to a symbol
   * This is because cudaMemcpyToSymbol is implicit local scope linkage. Meaning
   * cudaMemcpyToSymbol must be in the same generated .obj file of the kernel
   * where you want to use it. Link to more info below.
   * http://stackoverflow.com/questions/16997611/cuda-writing-to-constant-memory-wrong-value */
  // Copy struct best_2opt zero to GPU
  setParam(zero);

  // Determine thread size and block size
  gridSize = (num_cities + threadsPerBlock - 1) / threadsPerBlock;
  
  printf("blockSize = %d\n", threadsPerBlock);
  printf("gridSize = %d\n\n", gridSize);
  
  //to calculate the number of jobs/2-opt changes and iteration number for each thread
  unsigned long long counter = (long)(num_cities-2)*(long)(num_cities-1)/2;
  unsigned int iterations = (counter/(threadsPerBlock*gridSize)) + 1;

  // Execute kernel
  find_route<<<gridSize, threadsPerBlock>>>(num_cities, d_coords, threadsPerBlock, iterations);
  
  // Sync Device
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy from GPU to CPU
//  HANDLE_ERROR(cudaMemcpy(route, d_route, (num_cities+1) * sizeof(int), cudaMemcpyDeviceToHost));

  /* This is a wrapper function which allows the wrapper file to copy to a symbol
   * This is because cudaMemcpyToSymbol is implicit local scope linkage. Meaning
   * cudaMemcpyToSymbol must be in the same generated .obj file of the kernel
   * where you want to use it. Link to more info below.
   * http://stackoverflow.com/questions/16997611/cuda-writing-to-constant-memory-wrong-value */
  // Copy struct best_2opt to out from GPU
  getParam(&out);

  // Perform 2_opt swap and get new route
  // Must double check
//  if (out.i > out.j) {
//	  // Do something
//	  // swap(i, j) MAYBE
//  }

  // Calculate new distance from new route


  // Check if distance is less than old best distance



  // Continue while best_change is less than 0
  // WHY

  printf("out = %d, %d, %d\n", out.i, out.j, out.minchange);

  for (int i=0; i<num_cities+1; i++) {
    printf("%d ", route[i]);
  }
  printf("\n\n");

  cudaDeviceReset();
}
