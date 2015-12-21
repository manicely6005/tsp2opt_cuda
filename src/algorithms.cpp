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
 * algorithms.cpp
 *
 * Import TSPLIB file and call GPU wrapper file. Wrapper will call CUDA kernel
 * that runs the 2-opt.
 *
 ******************************************************************************/

#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <ctime>
#include "algorithms.h"
#include "edge_weight.h"
#include "wrapper.cuh"

struct tsp_info {
  std::string name;
  std::string type;
  int dim;
  std::string e_weight;
  int solution;
} tsp_info;

//Constructor that takes a character string corresponding to an external filename and reads in cities from that file
tsp::tsp(int argc, char * argv[])
{
  inputCoords = new struct city_coords [MAX_CITIES*2];
  orderCoords = new struct city_coords [MAX_CITIES*2];
  gpuResult = new struct best_2opt;

  num_cities = read_file(argc, argv);
  route = new int [num_cities+1];
  new_route = new int [num_cities+1];
  temp_route = new int [num_cities+1];

  if (tsp_info.e_weight.c_str() == pStr[0]) {
      pFun = &edge_weight::euc2d;

  } else if (tsp_info.e_weight.c_str() == pStr[1]) {
      pFun = &edge_weight::geo;

  } else if (tsp_info.e_weight.c_str() == pStr[2]) {
      pFun = &edge_weight::att;

  } else if (tsp_info.e_weight.c_str() == pStr[3]) {
      pFun = &edge_weight::ceil2d;

  } else {
      printf("Error calculating distance\n");
      exit(EXIT_FAILURE);
  }
}

//Destructor clears the deques
tsp::~tsp()
{
  delete(inputCoords);
  delete(orderCoords);
  delete(gpuResult);
  delete(route);
  delete(new_route);
  delete(temp_route);
}

//Read city data from file into tsp's solution member, returns number of cities added
int tsp::read_file(int argc, char *argv[])
{
  const char *filename;
  std::string input;

  if (argc==2) {
      (void)(input);
      filename = argv[1];
  } else {
      if (argc==1) {
	  printf("Enter inputfile ('TSPLIB/{$FILENAME}')\n");
	  std::cin >> input;
	  filename = input.c_str();
      } else {
	  printf("usage: tsp_2opt <input file (*.tsp)>");
	  exit(1);
      }
  }

  if  (!strcmp 		(filename,"TSPLIB/berlin52.tsp")) 	tsp_info.solution = 7542;
  else if (!strcmp 	(filename,"TSPLIB/ch130.tsp")) 		tsp_info.solution = 6110;
  else if (!strcmp 	(filename,"TSPLIB/pr439.tsp")) 		tsp_info.solution = 107217;
  else if (!strcmp 	(filename,"TSPLIB/kroA100.tsp")) 	tsp_info.solution = 21282;
  else if (!strcmp 	(filename,"TSPLIB/kroE100.tsp")) 	tsp_info.solution = 22068;
  else if (!strcmp 	(filename,"TSPLIB/kroB100.tsp")) 	tsp_info.solution = 22141;
  else if (!strcmp 	(filename,"TSPLIB/kroC100.tsp")) 	tsp_info.solution = 20749;
  else if (!strcmp 	(filename,"TSPLIB/kroD100.tsp")) 	tsp_info.solution = 21294;
  else if (!strcmp 	(filename,"TSPLIB/kroA150.tsp")) 	tsp_info.solution = 26524;
  else if (!strcmp 	(filename,"TSPLIB/kroA200.tsp")) 	tsp_info.solution = 29368;
  else if (!strcmp 	(filename,"TSPLIB/ch150.tsp")) 		tsp_info.solution = 6528;
  else if (!strcmp 	(filename,"TSPLIB/rat195.tsp")) 	tsp_info.solution = 2323;
  else if (!strcmp 	(filename,"TSPLIB/ts225.tsp")) 		tsp_info.solution = 126643;
  else if (!strcmp 	(filename,"TSPLIB/pr226.tsp")) 		tsp_info.solution = 80369;
  else if (!strcmp 	(filename,"TSPLIB/pr264.tsp")) 		tsp_info.solution = 49135;
  else if (!strcmp 	(filename,"TSPLIB/pr299.tsp")) 		tsp_info.solution = 48191;
  else if (!strcmp 	(filename,"TSPLIB/a280.tsp")) 		tsp_info.solution = 2579;
  else if (!strcmp 	(filename,"TSPLIB/att532.tsp")) 	tsp_info.solution = 27686;
  else if (!strcmp 	(filename,"TSPLIB/rat783.tsp")) 	tsp_info.solution = 8806;
  else if (!strcmp 	(filename,"TSPLIB/pr1002.tsp")) 	tsp_info.solution = 259045;
  else if (!strcmp 	(filename,"TSPLIB/vm1084.tsp")) 	tsp_info.solution = 239297;
  else if (!strcmp 	(filename,"TSPLIB/pr2392.tsp")) 	tsp_info.solution = 378032;
  else if (!strcmp 	(filename,"TSPLIB/fl3795.tsp")) 	tsp_info.solution = 28772;
  else if (!strcmp 	(filename,"TSPLIB/pcb3038.tsp")) 	tsp_info.solution = 137694;
  else if (!strcmp 	(filename,"TSPLIB/fnl4461.tsp")) 	tsp_info.solution = 182566;
  else if (!strcmp 	(filename,"TSPLIB/rl5934.tsp")) 	tsp_info.solution = 556045;
  else if (!strcmp 	(filename,"TSPLIB/pla7397.tsp")) 	tsp_info.solution = 23260728;
  else if (!strcmp 	(filename,"TSPLIB/usa13509.tsp")) 	tsp_info.solution = 19982859;
  else if (!strcmp 	(filename,"TSPLIB/d15112.tsp")) 	tsp_info.solution = 1573084;
  else if (!strcmp 	(filename,"TSPLIB/usa15309.tsp")) 	tsp_info.solution = 19982859;
  else if (!strcmp 	(filename,"TSPLIB/d18512.tsp")) 	tsp_info.solution = 645238;
  else if (!strcmp 	(filename,"TSPLIB/sw24978.tsp")) 	tsp_info.solution = 855597;
  else if (!strcmp 	(filename,"TSPLIB/pla33810.tsp")) 	tsp_info.solution = 66048945;
  else if (!strcmp 	(filename,"TSPLIB/pla85900.tsp")) 	tsp_info.solution = 142382641;
  else if (!strcmp 	(filename,"TSPLIB/mona-lisa100K.tsp"))	tsp_info.solution = 5757080;
  else tsp_info.solution = 0;

  // open file
  std::ifstream inputFile(filename);

  // Store each line
  std::string line;

  // Try to open file
  if (inputFile.is_open()) {
      while (!inputFile.eof()) {
	  getline(inputFile, line);

	  std::istringstream iss(line);
	  std::string result;

	  if (!line.find("NAME")) {
	      std::getline(iss, result, ':');
	      std::getline(iss, result, '\n');
	      result.erase(remove_if(result.begin(), result.end(), isspace), result.end());
	      tsp_info.name = result;

	  }	else if (!line.find("TYPE")) {
	      std::getline(iss, result, ':');
	      std::getline(iss, result, '\n');
	      result.erase(remove_if(result.begin(), result.end(), isspace), result.end());
	      tsp_info.type = result;

	  } else if (!line.find("DIMENSION")) {
	      std::getline(iss, result, ':');
	      std::getline(iss, result, '\n');
	      tsp_info.dim = stoi(result);

	  } else if (!line.find("EDGE_WEIGHT_TYPE")) {
	      std::getline(iss, result, ':');
	      std::getline(iss, result, '\n');
	      result.erase(remove_if(result.begin(), result.end(), isspace), result.end());
	      tsp_info.e_weight = result;

	  }  else {
	      std::istringstream in(line);
	      try {
		  std::copy(std::istream_iterator<float> (in), std::istream_iterator<float>(), std::back_inserter(coord));
	      } catch (std::exception const & e) {
		  // Empty
	      }
	  }
      }
  } else {
      printf("Unable to open file\n");
      exit(EXIT_FAILURE);
  }

  // close file
  inputFile.close();

  // copy coordinates to structure
  for (int i=0; i<tsp_info.dim; i++) {
      inputCoords[i].x = coord[i*3+1];
      inputCoords[i].y = coord[i*3+2];
  }

  printf("Name = %s\n", tsp_info.name.c_str());
  printf("Type = %s\n", tsp_info.type.c_str());
  printf("Dimension = %d\n", tsp_info.dim);
  printf("Edge Weight = %s\n", tsp_info.e_weight.c_str());
  printf("Solution = %d\n", tsp_info.solution);
  printf("\n");

  return tsp_info.dim;
}

void tsp::two_opt()
{
  int *temp;
  int distance, new_distance;
  bool improve = true;

  // Start timer
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  // Calculate initial route
  init_route();

  // Create ordered coordinates
  creatOrderCoord(route);

  // Check GPU Info
  getGPU_Info();

  printf("Optimal Distance = %d\n\n", tsp_info.solution);

  // Calculate initial route distance
  distance = (obj.*pFun)(num_cities, orderCoords);
  printf("Initial distance = %d\n\n", distance);

  // Initialize random seed
  srand(time(NULL));

  while(improve) {
      improve = false;

      cuda_function(num_cities, orderCoords, gpuResult);

      // Create new route with GPU swap result
      if (gpuResult->i < gpuResult->j) {
	  swap_two(gpuResult->i-1, gpuResult->j);
      } else {
	  swap_two(gpuResult->j, gpuResult->i-1);
      }

      // Create ordered coordinates from new route
      creatOrderCoord(new_route);

      // Calculate new distance
      new_distance = (obj.*pFun)(num_cities, orderCoords);
      //      printf("New_distance = %d\n\n", new_distance);

      // Check if new route distance is better than last best distance
      if (new_distance < distance) {
	  distance = new_distance;
	  temp = route;
	  route = new_route;
	  new_route = temp;
	  improve = true;

	  // If new distance is not less than the old but greater than desired
	  // This help find global minimum
      } else if (new_distance > (tsp_info.solution*1.05)) {
	  int ii = rand() % (num_cities - 1) + 1;	// Index range 1 to num_cities
	  int jj = rand() % (num_cities - 1) + 1;	// Can't select first or last index
	  if (ii < jj) {
	      swap_two(ii, jj);
	  } else {
	      swap_two(jj, ii);
	  }

	  // Create ordered coordinates from new route
	  creatOrderCoord(new_route);

	  // Calculate initial route distance
	  distance = (obj.*pFun)(num_cities, orderCoords);

	  temp = route;
	  route = new_route;
	  new_route = temp;
	  improve = true;
      }
  }

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  printf("Optimized Distance = %d\n\n", new_distance);

  printf("Optimal Distance = %d\n\n", tsp_info.solution);

  std::cout << "finished computation at " << std::ctime(&end_time)
  << "elapsed time: " << elapsed_seconds.count() << "s\n\n";

  resetGPU();
}

void tsp::swap_two(const int& i, const int& j)
{
  int count = 0;

  for (int c=0; c<i; c++) {
      new_route[count] = route[c];
      count++;
  }

  for (int c=j; c>i-1; c--) {
      new_route[count] = route[c];
      count++;
  }

  for (int c=j+1; c<num_cities+1; c++) {
      new_route[count] = route[c];
      count++;
  }
}

void tsp::init_route()
{
  for (int i=0; i<num_cities; i++) {
      route[i] = i+1;
  }
  route[num_cities] = 1;      // Return to beginning of tour
}

void tsp::print(int *arr)
{
  for (int i=0; i<num_cities+1; i++) {
      printf("%d ", arr[i]);
  }
  printf("\n\n");
}

void tsp::creatOrderCoord(int *arr) {
  for (int i=0; i<num_cities+1; i++) {
      orderCoords[i] = inputCoords[arr[i]-1];
  }
}
