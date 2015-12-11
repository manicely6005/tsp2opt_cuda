//
//  cuda.cuh
//
//
//  Created by Nicely, Matthew A on 10/23/15.
//
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// __global__ void find_route(int *route, int N, float *crap, int *matrix);

__device__ int count = 1; 
__device__ int jump = 2;
__device__ bool improve = true;

// __device__ void swap_two(int idx, int i, int j, int *route, int *matrix, int N);
// 
// __device__ void euc2d(int idx, int *matrix, int num_cities, float *crap, int *distance);
// __device__ void ceil2d(int idx, int *matrix, int num_cities, float *crap, int *distance);
// __device__ void geo(int idx, int *matrix, int num_cities, float *crap, int *distance);
// __device__ void att(int idx, int *matrix, int num_cities, float *crap, int *distance);