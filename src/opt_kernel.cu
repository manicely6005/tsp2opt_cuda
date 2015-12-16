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
 * opt_kernel.cu
 *
 * CUDA kernel to calculate 2-opt swap on GPU.
 *
 ******************************************************************************/

#include <stdio.h>
#include "opt_kernel.cuh"
#include "wrapper.cuh"
#include "algorithms.h"

#define WIDTH 3

__global__ void find_route(int num_cities,  city_coords *coords, unsigned long long counter, unsigned int iterations) {

   __shared__ city_coords cache[MAX_CITIES];
   __shared__ int cities;
   __shared__ best_2opt best_values[threadsPerBlock];

   register int idx = threadIdx.x + blockIdx.x * blockDim.x;
   register int id;
   register unsigned int i, j;
   register unsigned long long max = counter;
   register int packSize = blockDim.x * gridDim.x;
   register unsigned int iter = iterations;
   register int change;
   register int minChange = 999999;

   cities = num_cities;

   for (register int i=threadIdx.x; i<cities; i+= blockDim.x) {
	   cache[i] = coords[i];
   }

   __syncthreads();

   // Each thread performs iter inner iterations in order to reuse the shared memory
   /* The following technique was taken from "Accelerating 2-opt and 3-opt Local Search
    * Using GPU in the Travelling Salesman Problem" by Kamil Rocki
    */

   for (register int no = 0; no < iter; no++) {
	   id = idx + no * packSize;

	   if (id < max) {
		   // Indexing Lower Triangular Matrix
		   i = (unsigned int)(3 +__fsqrt_rn(8.0f * (float) id + 1.0f))/2;
		   j = id - (i-2) * (i-1) / 2 + 1;

		   // Calculate change
//		   change =

		   // Save if local thread change is better then previous iteration best
//		   if (change < minChange) {
//			 best_value[threadIdx.x].minchange = change;
//			 best_value[threadIdx.x].i = i;
//			 best_value[threadIdx.x].j = j;
//		   }
	   }
   }

   __syncthreads();

   // Intra-block reduction
   // Reductions, threadsPerBlock must be a power of 2 because of the following code
//     register int k = blockDim.x/2;
//     while (k != 0) {
//       if (threadIdx.x < i) {
//         if (best_values[threadIdx.x + k].minchange < best_values[threadIdx.x].minchange) {
//			 	best_values[threadIdx.x].minchange = best_values[threadIdx.x + k].minchange;
//   			best_values[threadIdx.x].i = best_values[threadIdx.x + k].i;
//   			best_values[threadIdx.x].j = best_values[threadIdx.x + k].j;
//         }
//       }
//       __syncthreads();
//       i /= 2;
//     }

   __syncthreads();

   // Inter-block reduction. This will be a serial process of all blocks... Might be faster to reduce number of blocks...
   if (idx == 0) {

//	   if (best_values[threadIdx.x].minchange < best.minchange) {
//		   // Atomic function
//		   atomicMin(&(best.minchange), best_values[threadIdx.x].minchange)
//	   }

	   printf("best = %d, %d, %d\n", best.i, best.j, best.minchange);\
	   best.i = best.i + 5;
	   best.minchange  = 1000;

	   for (int i=0; i<cities; i++) {
		   printf("cache[i].x = %f ", cache[i].x);
	       printf("cache[i].y = %f\n", cache[i].y);
	   }
   }
}

/* This is a wrapper function which allows the wrapper file to copy to a symbol
 * This is because cudaMemcpyToSymbol is implicit local scope linkage. Meaning
 * cudaMemcpyToSymbol must be in the same generated .obj file of the kernel
 * where you want to use it. Link to more info below.
 * http://stackoverflow.com/questions/16997611/cuda-writing-to-constant-memory-wrong-value */
__host__ void setParam(struct best_2opt zero) {
	cudaMemcpyToSymbol(best, &zero, sizeof(struct best_2opt));
}

/* This is a wrapper function which allows the wrapper file to copy to a symbol
 * This is because cudaMemcpyToSymbol is implicit local scope linkage. Meaning
 * cudaMemcpyToSymbol must be in the same generated .obj file of the kernel
 * where you want to use it. Link to more info below.
 * http://stackoverflow.com/questions/16997611/cuda-writing-to-constant-memory-wrong-value */
__host__ void getParam(struct best_2opt * out) {
	cudaMemcpyFromSymbol(out, best, sizeof(struct best_2opt));
}

// __device__ void geo(int idx, int *matrix, int num_cities, float *crap, int *distance) {
//   
//   int deg, j;
//   double xi, yi, xj, yj;
//   double PI = 3.141492;
//   double min, latitude_i, latitude_j, longitude_i, longitude_j, RRR, q1, q2, q3;
//   
//   for (int i=0; i<num_cities; i++) {
//     j = i + 1;
//     
//     // matrix[i] - 1 convert the 1 based matrix to the 0 based crap
//     xi = crap[(matrix[idx*(num_cities+1)+i] - 1)*WIDTH+1];    // x coordinate
//     yi = crap[(matrix[idx*(num_cities+1)+i] - 1)*WIDTH+2];    // y coordinate
//     xj = crap[(matrix[idx*(num_cities+1)+j] - 1)*WIDTH+1];    // x coordinate
//     yj = crap[(matrix[idx*(num_cities+1)+j] - 1)*WIDTH+2];    // y coordinate
//     
//     //     printf("xi = %f : yi = %f : xj = %f : yj = %f\n", xi, yi, xj, yj);
//     
//     deg = (int) xi;
//     min = xi - deg;
//     latitude_i = PI * (deg + 5.0 * min/3.0)                                                                 / 180.0;
//     
//     deg = (int) yi;
//     min = yi - deg;
//     longitude_i = PI * (deg + 5.0 * min/3.0) / 180.0;
//     
//     deg = (int) xj;
//     min = xj - deg;
//     latitude_j = PI * (deg + 5.0 * min/3.0) / 180.0;
//     
//     deg = (int) yj;
//     min = yj - deg;
//     longitude_j = PI * (deg + 5.0 * min/3.0) / 180.0;
//     
//     // The distance between two different nodes i and j in kilometers is then computed as follows:
//     RRR = 6378.388;
//     
//     q1 = cos(longitude_i - longitude_j);
//     q2 = cos(latitude_i - latitude_j);
//     q3 = cos(latitude_i + latitude_j);
//     
//     *distance += (int) (RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
//   }
// }
