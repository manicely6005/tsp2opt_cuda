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

__global__ __launch_bounds__(1024,4)
void find_route(int num_cities,  city_coords *coords, unsigned long long counter, int iterations, best_2opt *best_block) {

  // Dynamically allocated shared memory
  extern __shared__ best_2opt s[];
  best_2opt *best_thread = s;
  city_coords *cache = (city_coords*)&best_thread[threadsPerBlock];

  register unsigned int tid = threadIdx.x;
  register unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  register unsigned int id;
  register int i, j;
  register unsigned long long max = counter;
  register unsigned int packSize = blockDim.x * gridDim.x;
  register int iter = iterations;
  register int change;
  register struct best_2opt best = {0,0,999999};
  register int cities = num_cities;

  for (register int i=tid; i<cities; i+= blockDim.x) {
    cache[i] = coords[i];
  }
  __syncthreads();

  // Each thread performs iter inner iterations in order to reuse the shared memory
  /* The following for loop technique was taken from "Accelerating 2-opt and 3-opt Local Search
   * Using GPU in the Traveling Salesman Problem" by Kamil Rocki
   */

#pragma unroll
  for (register int no = 0; no < iter; no++) {
    id = idx + no * packSize;

    if (id < max) {
      // Indexing Lower Triangular Matrix i = (3 + sqrt(8 * id + 1)) / 2
      i = (int) (3 + __fsqrt_rn(__fmul_rn((float) 8.0, __fadd_rn((float) id, (float) 1.0)))) >> 1;
      j = id - (i-2) * (i-1) / 2 + 1;

      // Calculate change
      change = euc2d(i, j, cache) + euc2d(i-1, j-1, cache) - euc2d(i-1, i, cache) - euc2d(j-1, j, cache);

      // Save if local thread change is better then previous iteration best
      if (change < best.minchange) {
        best.minchange = change;
        best.i = i;
        best.j = j;
        best_thread[tid] = best;
      }
    }
  }
  __syncthreads();

  // Intra-block reduction
  // Reductions, threadsPerBlock must be a power of 2 because of the following code
  register unsigned int k = blockDim.x >> 1;
  while (k != 32) {
    if (threadIdx.x < k) {
      best_thread[tid] = (best_thread[tid + k].minchange < best_thread[tid].minchange) ? best_thread[tid + k]:best_thread[tid];
    }
    __syncthreads();
    k >>= 1;
  }

  // No reason to sync a single warp
  if (threadIdx.x <= 32) {
    best_thread[tid] = (best_thread[tid + 32].minchange < best_thread[tid].minchange) ? best_thread[tid + 32]:best_thread[tid];
    best_thread[tid] = (best_thread[tid + 16].minchange < best_thread[tid].minchange) ? best_thread[tid + 16]:best_thread[tid];
    best_thread[tid] = (best_thread[tid + 8].minchange < best_thread[tid].minchange) ? best_thread[tid + 8]:best_thread[tid];
    best_thread[tid] = (best_thread[tid + 4].minchange < best_thread[tid].minchange) ? best_thread[tid + 4]:best_thread[tid];
    best_thread[tid] = (best_thread[tid + 2].minchange < best_thread[tid].minchange) ? best_thread[tid + 2]:best_thread[tid];
    best_thread[tid] = (best_thread[tid + 1].minchange < best_thread[tid].minchange) ? best_thread[tid + 1]:best_thread[tid];
  }

  // Copying best match from each block to global memory. Reduction will be performed on CPU
  if (threadIdx.x == 0) {
    best_block[blockIdx.x] = best_thread[threadIdx.x];
  }
}

__inline__ __device__ int euc2d(int i, int j, struct city_coords *coords) {
  register float xi, yi, xj, yj, xd, yd;

  xi = coords[i].x;
  yi = coords[i].y;
  xj = coords[j].x;
  yj = coords[j].y;

  xd = (xi - xj) * (xi - xj);
  yd = (yi - yj) * (yi - yj);

  return ((int) floor(sqrtf(xd + yd) + 0.5));
}

//__device__ int geo(int i, int j, struct city_coords *coords) {
//  register int deg;
//  register float xi, yi, xj, yj;
//  register double PI = 3.141492;
//  register double min, latitude_i, latitude_j, longitude_i, longitude_j, RRR, q1, q2, q3;
//
//  xi = coords[i].x;
//  yi = coords[i].y;
//  xj = coords[j].x;
//  yj = coords[j].y;
//
//  deg = (int) xi;
//  min = xi - deg;
//  latitude_i = PI * (deg + 5.0 * min/3.0) / 180.0;
//
//  deg = (int) yi;
//  min = yi - deg;
//  longitude_i = PI * (deg + 5.0 * min/3.0) / 180.0;
//
//  deg = (int) xj;
//  min = xj - deg;
//  latitude_j = PI * (deg + 5.0 * min/3.0) / 180.0;
//
//  deg = (int) yj;
//  min = yj - deg;
//  longitude_j = PI * (deg + 5.0 * min/3.0) / 180.0;
//
//  // The distance between two different nodes i and j in kilometers is then computed as follows:
//  RRR = 6378.388;
//
//  q1 = cos(longitude_i - longitude_j);
//  q2 = cos(latitude_i - latitude_j);
//  q3 = cos(latitude_i + latitude_j);
//
//  return ((int) (RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0));
//}


