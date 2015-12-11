//
//  cuda.cu
//
//
//  Created by Nicely, Matthew A on 10/23/15.
//
//
#include <stdio.h>
#include "opt_kernel.cuh"
#include "algorithms.h"

#define CRAP 15
#define WIDTH 3



// __global__ void find_route(int *route, int num_cities, float *crap, int *matrix) {
//   
//   __shared__ int cache[threadsPerBlock];
//   
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   
//   int *temp;
//   int distance, i, j;
//   int swaps = num_cities - 2;
//   
//   //   bool improve = true;
//   bool jump = false;
//   
//   while (idx < num_cities) {
// //     if(idx == 0) {
// //       printf("hello\n");
// //       for (int i=0; i<num_cities+1; i++) {
// // 	printf("%d ", route[i]);
// //       }
// //       printf("\n");
// //     }
//     
//     while (improve) {
//       improve = false;
//       jump = false;
//       
//       for (int a=0; a<swaps; a++) {
// 	
// 	if (jump) break;
// 	
// 	if (idx < (swaps - a)) {
// 	  distance = 0;
// 	  i = a+1;
// 	  j = idx + a+2;
// 	  
// 	  if(idx == 0) {
// 	    printf("hello1\n");
// 	    for (int i=0; i<num_cities+1; i++) {
// 	      printf("%d ", route[i]);
// 	    }
// 	    printf("\n");
// 	  }
// 	  
// 	  // Execute swap two algorithm
// 	  swap_two(idx, i, j, route, matrix, num_cities);
// 	  
// 	  if(idx == 0) {
// 	    printf("hello2\n");
// 	    for (int i=0; i<num_cities+1; i++) {
// 	      printf("%d ", route[i]);
// 	    }
// 	    printf("\n");
// 	  }
// 	  
// 	  // Calculate distance of new route
// 	  att(idx, matrix, num_cities, crap, &distance);
// 	  
// 	  // Determine if new distance is shorter than original
// 	  if (distance < d_distance) {
// 	    cache[threadIdx.x] = distance;
// 	  }
// 	}
// 	
// 	if(idx==0) {
// 	  printf("this\n");
// 	  for (int i=0; i<num_cities+1; i++) {
// 	    printf("%d ", cache[i]);
// 	  }
// 	  printf("\n");
// 	}
// 	
// 	__syncthreads();
// 	
// 	if (idx==0) {
// 	  for (int i=0; i<threadsPerBlock; i++) {
// 	    if ((cache[i] > 0) && (cache[i] < d_distance)) {
// 	      d_distance = cache[i];
// 	      	      printf("New distance = %d\n", d_distance);
// 	      	      printf("New route @ %d\n", i);
// 	      	      for (int j=0; j<(num_cities+1); j++) {
// 	      		printf("%d ", matrix[i*15+j]);
// 	      	      }
// 	      	      printf("\n\n");
// 	      
// 	      memcpy(route, matrix + (i*(num_cities+1)+0), (num_cities + 1) * sizeof(int));
// 	      
// 	      jump = true;
// 	      improve = true;
// 	      printf("New route\n");
// 	      for (int j=0; j<(num_cities+1); j++) {
// 		printf("%d ", route[j]);
// 	      }
// 	      printf("\n\n");
// 	    }
// 	  }
// 	}
// 	cache[threadIdx.x] = 0;	// Set shared memory to 0
//       }
//     }
//     
//     // Increment idx to exit while loop
//     idx += num_cities + 1;
//   }
// }

// __device__ void swap_two(int idx, int i, int j, int *route, int *matrix, int num_cities) {
//   int count = 0;
//   
//   for (int c=0; c<i; c++) {
//     matrix[idx*(num_cities+1)+count] = route[c];
//     count++;
//   }
//   
//   for (int c=j; c>i-1; c--) {
//     matrix[idx*(num_cities+1)+count] = route[c];
//     count++;
//   }
//   
//   for (int c=j+1; c<num_cities+1; c++) {
//     matrix[idx*(num_cities+1)+count] = route[c];
//     count++;
//   }
// }
// 
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
// 
// __device__ void att(int idx, int *matrix, int num_cities, float *crap, int *distance) {
//   int dij, tij, j, xi, xj, yi, yj, xd, yd;
//   float rij;
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
//     xd = pow(double(xi - xj), 2.0);
//     yd = pow(double(yi - yj), 2.0);
//     
//     rij = sqrt((xd + yd) / 10.0);
//     
//     tij = round(rij);
//     
//     if (tij < rij) {
//       dij = tij + 1;
//     } else {
//       dij = tij;
//     }
//     *distance += dij;
//   }
// }
