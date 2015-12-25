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
 * opt_kernel.cuh
 *
 ******************************************************************************/

#ifndef OPT_KERNEL_CUH
#define OPT_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "algorithms.h"

__global__ void find_route(int N, city_coords *coords, unsigned long long counter, unsigned int iterations, best_2opt *best_block);

__device__ best_2opt best;

__device__ int euc2d(int i, int j, struct city_coords *coords);
//__device__ int ceil2d(int i, int j, struct city_coords *coords);
//__device__ int geo(int i, int j, struct city_coords *coords);
//__device__ int att(int i, int j, struct city_coords *coords);

#endif
