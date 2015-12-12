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
 * algorithms.h
 *
 ******************************************************************************/

#ifndef _algorithms_h
#define _algorithms_h

#include <vector>
#include <string>
#include "edge_weight.h"

//maximum number of cities that can be used in the simple GPU 2-opt algorithm
//limited by the shared memory size
//shared memory needed = MAX_CITIES * sizeof(city_coords)
#define MAX_CITIES 4096

struct city_coords{
  float x;
  float y;
};

class tsp
{
	
public:
  tsp(int argc, char *argv[]); // Constructor, takes filename to read from as input
  tsp(tsp & source);
  ~tsp();
  int read_file(int argc, char *argv[], struct city_coords *coords); // Reads a list of cities into original_list from filename
  int dist(int i, int j); // Calculates the Euclidean distance to another city
  void two_opt(); // Attempt at 2-opt
  void swap_two(const int& i, const int& j); // Used by two_opt()
  void init_route(); // Calculate initial route
  void print(int *arr);
	
private:
  const std::string pStr[4] = {"EUC_2D", "GEO", "ATT", "CEIL_2D"};
  int num_cities; // Stores the number of cities read into original_list
  std::vector<float> coord;
  city_coords *coords;
  int *route;
  int *new_route;
  int *temp_route;
//   int (edge_weight::*pFun) (int *arr, int num_cities, std::vector<float>&coord);
  int (edge_weight::*pFun) (int *arr, int num_cities, struct city_coords *coords);
  edge_weight obj;
};
#endif
