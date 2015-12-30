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

#include <iostream>
#include <sstream>
#include <iterator>
#include <cstring>
#include <algorithm>
#include <string>
#include <ctime>
#include "algorithms.h"
#include "edge_weight.h"
#include "wrapper.cuh"

// Global
std::ofstream myfile;

// Continue search if true
volatile sig_atomic_t improve;

// Function to handle alarm signal
void handle_alarm(int) {
  improve = false;
  alarm(timeLimit);
}

// Constructor that takes a character string corresponding to an external filename and reads in cities from that file
tsp::tsp(int argc, char * argv[]) {
  inputCoords = new struct city_coords [maxCities*2];
  orderCoords = new struct city_coords [maxCities*2];
  gpuResult = new struct best_2opt;
  tsp_info = new struct tsp_info;

  num_cities = read_file(argc, argv);
  route = new int [num_cities+1];
  new_route = new int [num_cities+1];
  temp_route = new int [num_cities+1];

  if (tsp_info->e_weight.c_str() == pStr[0]) {
      pFun = &edge_weight::euc2d;
  } else if (tsp_info->e_weight.c_str() == pStr[1]) {
      pFun = &edge_weight::geo;
  } else if (tsp_info->e_weight.c_str() == pStr[2]) {
      pFun = &edge_weight::att;
  } else if (tsp_info->e_weight.c_str() == pStr[3]) {
      pFun = &edge_weight::ceil2d;
  } else {
      printf("Error calculating distance\n");
      exit(EXIT_FAILURE);
  }

  // Create filename with timestamp
  std::ostringstream oss;
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  char buffer [80];
  strftime(buffer, 80, "_%F_%H-%M-%S", timeinfo);
  oss << "results/" << tsp_info->name << buffer << ".txt";

  // Open output file
  myfile.open(oss.str(), std::ofstream::out | std::ofstream::app);
  if (!myfile.good()) {
      printf("Couldn't open output file.");
      exit(EXIT_FAILURE);
  }
}

// Destructor clears the deques
tsp::~tsp(void) {
  delete(inputCoords);
  delete(orderCoords);
  delete(gpuResult);
  delete(tsp_info);
  delete(route);
  delete(new_route);
  delete(temp_route);

  // Close output file
  myfile.close();
}

// Read city data from file into tsp's solution member, returns number of cities added
int tsp::read_file(int argc, char *argv[]) {
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
	  exit(EXIT_FAILURE);
      }
  }

  if  (!strcmp 		(filename,"TSPLIB/a280.tsp")) 		tsp_info->solution = 2579;
  else if (!strcmp 	(filename,"TSPLIB/att48.tsp")) 		tsp_info->solution = 10628;
  else if (!strcmp 	(filename,"TSPLIB/att532.tsp")) 	tsp_info->solution = 27686;
  else if (!strcmp 	(filename,"TSPLIB/bier127.tsp")) 	tsp_info->solution = 118282;
  else if (!strcmp 	(filename,"TSPLIB/burma14.tsp")) 	tsp_info->solution = 3323;
  else if (!strcmp 	(filename,"TSPLIB/d15112.tsp")) 	tsp_info->solution = 1573084;
  else if (!strcmp 	(filename,"TSPLIB/d1655.tsp")) 		tsp_info->solution = 62128;
  else if (!strcmp 	(filename,"TSPLIB/d2103.tsp")) 		tsp_info->solution = 80450;
  else if (!strcmp 	(filename,"TSPLIB/d493.tsp")) 		tsp_info->solution = 35002;
  else if (!strcmp 	(filename,"TSPLIB/d657.tsp")) 		tsp_info->solution = 48912;
  else if (!strcmp 	(filename,"TSPLIB/eil76.tsp")) 		tsp_info->solution = 538;
  else if (!strcmp 	(filename,"TSPLIB/fl3795.tsp")) 	tsp_info->solution = 28772;
  else if (!strcmp 	(filename,"TSPLIB/pcb3038.tsp")) 	tsp_info->solution = 137694;
  else if (!strcmp 	(filename,"TSPLIB/pr2392.tsp")) 	tsp_info->solution = 378032;
  else if (!strcmp 	(filename,"TSPLIB/rd400.tsp")) 		tsp_info->solution = 15281;
  else if (!strcmp 	(filename,"TSPLIB/rl11849.tsp")) 	tsp_info->solution = 923288;
  else if (!strcmp 	(filename,"TSPLIB/rl1889.tsp")) 	tsp_info->solution = 316536;
  else if (!strcmp 	(filename,"TSPLIB/rl5934.tsp")) 	tsp_info->solution = 556045;
  else if (!strcmp 	(filename,"TSPLIB/u2319.tsp")) 		tsp_info->solution = 234256;
  else tsp_info->solution = 0;

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
	      tsp_info->name = result;

	  }	else if (!line.find("TYPE")) {
	      std::getline(iss, result, ':');
	      std::getline(iss, result, '\n');
	      result.erase(remove_if(result.begin(), result.end(), isspace), result.end());
	      tsp_info->type = result;

	  } else if (!line.find("DIMENSION")) {
	      std::getline(iss, result, ':');
	      std::getline(iss, result, '\n');
	      tsp_info->dim = std::stoi(result);

	  } else if (!line.find("EDGE_WEIGHT_TYPE")) {
	      std::getline(iss, result, ':');
	      std::getline(iss, result, '\n');
	      result.erase(remove_if(result.begin(), result.end(), isspace), result.end());
	      tsp_info->e_weight = result;

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
  for (int i=0; i<tsp_info->dim; i++) {
      inputCoords[i].x = coord[i*3+1];
      inputCoords[i].y = coord[i*3+2];
  }

  printf("Name = %s\n", tsp_info->name.c_str());
  printf("Type = %s\n", tsp_info->type.c_str());
  printf("Dimension = %d\n", tsp_info->dim);
  printf("Edge Weight = %s\n", tsp_info->e_weight.c_str());
  printf("Solution = %d\n", tsp_info->solution);
  printf("\n");

  // Set tolerance
  tolerance = (tsp_info->dim < (int) 1000) ? lowTolerance:highTolerance;

  return (tsp_info->dim);
}

void tsp::two_opt(void) {
  // Set alarm to terminate problem once time limit is meet.
  set_alarm();

  // Initialize high resolution clock variables
  std::chrono::time_point<std::chrono::high_resolution_clock> start, middle, end;
  std::chrono::duration<double> elapsed_seconds;

  // Create objects
  wrapper wrapper(num_cities);

  for (int seed=0; seed<seedCount; seed++) {
      improve = true;	// Allow search if true (time limit has been reached).

      // Start timer
      start = std::chrono::high_resolution_clock::now();

      // Initialize random seed
      srand(seed);

      printf("Starting seed %d\n", seed);

      // Calculate initial route
      init_route();

      // Create ordered coordinates
      creatOrderCoord(route);

      // Calculate initial route distance
      distance = (obj.*pFun)(num_cities, orderCoords);
      printf("Initial distance = %d\n", distance);

      while(improve) {

	  // Call cuda wrapper
	  wrapper.cuda_function(num_cities, orderCoords, gpuResult);

	  // Create new route with GPU swap result
	  swap_two();

	  // Create ordered coordinates from new route
	  creatOrderCoord(new_route);

	  // Calculate new distance
	  new_distance = (obj.*pFun)(num_cities, orderCoords);

	  // Check if new route distance is better than last best distance
	  if (new_distance < distance) {
	      distance = new_distance;
	      replace_route();

	      // If new distance is greater than the old distance and
	      // the old distance is greater than the tolerance
	      // Swap two random edges and continue search
	  } else if (distance > (int)(tsp_info->solution * tolerance)) {
	      get_random();
	      swap_two();

	      // Create ordered coordinates from new route
	      creatOrderCoord(new_route);

	      // Calculate new route distance
	      distance = (obj.*pFun)(num_cities, orderCoords);

	      replace_route();
	  } else {
	      // The new distance is not better than old distance and the
	      // old distance meets the tolerance requirements
	      // Search complete
	      improve = false;
	  }
      }

      // Get seed runtime
      end = std::chrono::high_resolution_clock::now();
      elapsed_seconds = end-start;

      write_file(elapsed_seconds);
      printf("Optimized Distance = %d: Seed %d\n\n", new_distance, seed);
      printf("elapsed time: %f seconds\n\n", elapsed_seconds.count());
  }

  // Get system time
  std::chrono::system_clock::time_point endPoint = std::chrono::system_clock::now();
  std::time_t end_time = std::chrono::system_clock::to_time_t(endPoint);
  printf("finished computation at %s\n",std::ctime(&end_time));
}

void tsp::swap_two(void) {
  int count = 0;
  int c, i, j;

  // There's an issue, corrected with i
  if (gpuResult->i < gpuResult->j) {
      i = gpuResult->i-1;
      j = gpuResult->j;
  } else {
      i = gpuResult->j;
      j = gpuResult->i-1;
  }

  for (c=0; c<i; c++) {
      new_route[count] = route[c];
      count++;
  }

  for (c=j; c>i-1; c--) {
      new_route[count] = route[c];
      count++;
  }

  for (c=j+1; c<num_cities+1; c++) {
      new_route[count] = route[c];
      count++;
  }
}

void tsp::init_route(void) {
  for (int i=0; i<num_cities; i++) {
      route[i] = i+1;
  }
  route[num_cities] = 1;      // Return to beginning of tour
}

void tsp::print(int *arr) {
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

void tsp::write_file(std::chrono::duration<double> elapsed_seconds) {
  myfile << elapsed_seconds.count()<< ", " << distance << "\n";
}

void tsp::replace_route(void) {
  temp_route = route;
  route = new_route;
  new_route = temp_route;
}

void tsp::get_random(void) {
  gpuResult->i = rand() % (num_cities - 1) + 2;	// Index range 1 to num_cities
  gpuResult->j = rand() % (num_cities - 1) + 1;	// Can't select first or last index
}

void tsp::set_alarm(void) {
  // Initialize alarm to 5 seconds
    struct sigaction act;
    act.sa_handler = handle_alarm;
    act.sa_flags = SA_RESTART;
    sigaction(SIGALRM, &act, NULL);
    alarm(timeLimit);
}
