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

#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <signal.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include "edge_weight.h"


//maximum number of cities that can be used in the simple GPU 2-opt algorithm
//limited by the shared memory size
const int maxCities = 4096;
const int timeLimit = 1200;
const int seedCount = 50;
const float lowTolerance = 1.05; 	// Tolerance for data sets smaller than a thousand
const float highTolerance = 1.12;	// Tolerance for data sets larger than a thousand

struct city_coords{
  float x;
  float y;
};

struct best_2opt {
  int i;
  int j;
  int minchange;
};

struct tsp_info {
  std::string name;
  std::string type;
  int dim;
  std::string e_weight;
  unsigned int solution;
};

// File to hold results
extern std::ofstream myfile;

class tsp
{

public:
  tsp(int argc, char *argv[]); // Constructor, takes filename to read from as input
  tsp(tsp & source);
  ~tsp(void);
  int read_file(int argc, char *argv[]); // Reads a list of cities into original_list from filename
  void two_opt(void); // Attempt at 2-opt
  void swap_two(); // Used by two_opt()
  void init_route(void); // Calculate initial route
  void print(int *arr);
  void creatOrderCoord(int *arr);
  void write_file(std::chrono::duration<double> elapsed_seconds);
  void replace_route(void);
  void get_random(void);
  void set_alarm(void);
private:
  const std::string pStr[4] = {"EUC_2D", "GEO", "ATT", "CEIL_2D"};
  int num_cities, distance, new_distance;
  std::vector<float> coord;
  struct city_coords *inputCoords, *orderCoords;
  struct best_2opt *gpuResult;
  struct tsp_info *tsp_info;
  int *route, *new_route, *temp_route;
  int (edge_weight::*pFun) (int num_cities, struct city_coords *coords);  edge_weight obj;
  float tolerance;
};
#endif
