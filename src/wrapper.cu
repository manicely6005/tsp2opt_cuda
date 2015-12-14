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

//__global__ void find_route(int *route, int num_cities) ;
//{
//
//	printf("Here");
//	printf("Here");
//
//  __shared__ int cache[1024];
//
//  int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
////   if (idx < num_cities) {
//	   printf("idx = %d", idx);
////   }
//}

__device__ int d_distance;

__host__ void HandleError(cudaError_t err, const char *file, int line) {
	printf("cudaSuccess = %d and err = %d\n", cudaSuccess, err);
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),	file, line );
    exit( EXIT_FAILURE );
  }
} // HandleError
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int getGPU_Info(void) {
  
  int deviceCount = 0;
  
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  
  if (error_id != cudaSuccess) {
    printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
  }
  
  if (deviceCount == 0) {
    printf("There is no device supporting CUDA\n");
  }
  else {
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
    
    printf("Total amount of global memory: %.0f MBytes (%llu bytes)\n", 
	   (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);        
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
  
  return deviceCount;
}

// C++ CUDA Kernel wrapper
//void cuda_function(int *route, int *distance, int num_cities) {
//
//  int blockSize;      // The launch configurator returned block size
//  int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
//  int gridSize;       // The actual grid size needed, based on input size
//
//  // Create variables for GPU
//  int *d_route;
//
//  // Allocate memory on GPU
////  printf("Here0\n");
////  HANDLE_ERROR(cudaMalloc((void**)&d_route, (num_cities+1) * sizeof(int)));
//
//  // Memory set to zero
////  HANDLE_ERROR(cudaMemset(d_matrix, 0, (num_cities-2) * (num_cities+1) * sizeof(int)));
//
//  // Copy from CPU to GPU
////  printf("Here1\n");
////  HANDLE_ERROR(cudaMemcpy(d_route, route, (num_cities+1) * sizeof(int), cudaMemcpyHostToDevice));
////  HANDLE_ERROR(cudaMemcpyToSymbol(d_distance, distance, sizeof(int)));
//
//  // Determine thread size and block size
//  //   HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, find_route, 0, 0));
//  gridSize = (num_cities + threadsPerBlock - 1) / threadsPerBlock;
//
//  //   printf("minGridSize = %d\n", minGridSize);
//  printf("blockSize = %d\n", threadsPerBlock);
//  printf("gridSize = %d\n\n", gridSize);
//
//  // Execute kernel
//  find_route<<<gridSize, threadsPerBlock>>>(d_route, num_cities);
//
//  // Sync Device
////  printf("Here2\n");
//  HANDLE_ERROR(cudaDeviceSynchronize());
//
//  // Copy from GPU to CPU
////  printf("Here3\n");
////  HANDLE_ERROR(cudaMemcpy(route, d_route, (num_cities+1) * sizeof(int), cudaMemcpyDeviceToHost));
////  HANDLE_ERROR(cudaMemcpyFromSymbol(distance, d_distance, sizeof(int)));
//
//  for (int i=0; i<num_cities+1; i++) {
//    printf("%d ", route[i]);
//  }
//  printf("\n\n");
//
//  cudaDeviceReset();
//}

void cuda_function(int *route, int num_cities) {
  
  int blockSize;      // The launch  returned block size
  int gridSize;       // The actual grid size needed
  
  // Create variables for GPU
  int *d_route;
  
  // Allocate memory on GPU
  
  // Copy from CPU to GPU
  
  // Determine thread size and block size
  gridSize = (num_cities + threadsPerBlock - 1) / threadsPerBlock;
  
  printf("blockSize = %d\n", threadsPerBlock);
  printf("gridSize = %d\n\n", gridSize);
  
  // Execute kernel
  find_route<<<gridSize, threadsPerBlock>>>(d_route, num_cities);
  
  // Sync Device
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy from GPU to CPU
  
  for (int i=0; i<num_cities+1; i++) {
    printf("%d ", route[i]);
  }
  printf("\n\n");

  cudaDeviceReset();
}
