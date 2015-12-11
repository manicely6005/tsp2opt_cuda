#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include "algorithms.h"
#include "edge_weight.h"
#include "wrapper.cuh"

struct tsp_info {
  std::string name;
  std::string type;
  int dim;
  std::string e_weight;
} tsp_info;

//Constructor that takes a character string corresponding to an external filename and reads in cities from that file
tsp::tsp(int argc, char * argv[])
{
  coords = new struct city_coords [MAX_CITIES*2];
  
  num_cities = read_file(argc, argv, coords);
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
  delete(coords);
  delete(route);
  delete(new_route);
  delete(temp_route);
}

//Read city data from file into tsp's solution member, returns number of cities added
int tsp::read_file(int argc, char *argv[], struct city_coords *coords)
{
  const char *filename;
  std::string input;
  
  if (argc==2) {
    (void)(input);
    filename = argv[1];
  } else {
    if (argc==1) {
      printf("Enter inputfile\n");
      std::cin >> input;
      filename = input.c_str();
    } else {
      printf("usage: tsp_2opt <input file (*.tsp)>");
      exit(1);
    }
  }
  
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
    coords[i].x = coord[i*3+1];
    //     printf("coords[i].x = %f\n", coords[i].x);
    coords[i].y = coord[i*3+2];
    //     printf("coords[i].y = %f\n", coords[i].y);
  }
  
  printf("Name = %s\n", tsp_info.name.c_str());
  printf("Type = %s\n", tsp_info.type.c_str());
  printf("Dimension = %d\n", tsp_info.dim);
  printf("Edge Weight = %s\n", tsp_info.e_weight.c_str());
  printf("\n");
  
  return tsp_info.dim;
}

void tsp::two_opt()
{
  int *temp;
  int distance, new_distance, temp_distance;
  
  init_route();
  
  // Check GPU Info
  getGPU_Info();

  printf("Size of int = %lu\n", sizeof(int));
  printf("Size of float = %lu\n", sizeof(float));
  
  // Calculate initial route distance
  distance = (obj.*pFun)(route, num_cities, coords);
  printf("Initial distance = %d\n\n", distance);
  
  bool improve = true;
  bool update = false;
  bool jump;
  
  while(improve) {
    improve = false;
    jump = false;
    
    for (int i=1; i<num_cities-1; i++) {
      
      if(jump) break;
      temp_distance = distance;
      
      for (int j=i+1; j<num_cities; j++) {
	
	swap_two(i, j);
	// 	print(new_route);
	
	// Calculate new distance
	new_distance = (obj.*pFun)(new_route, num_cities, coords);
	// 	printf("Distance = %d\n", new_distance);
	if (new_distance < temp_distance) {
	  // 	  printf("New distance = %d\n", new_distance);
	  temp_distance = new_distance;
	  
	  temp = temp_route;
	  temp_route = new_route;
	  new_route = temp;
	  
	  improve = true;
	  jump = true;
	  update = true;
	}
      }
      if (update) {
	distance = temp_distance;
	temp = route;
	route = temp_route;
	temp_route = temp;
	update = false;
      }
    }
  }
  
  //  printf("Optimized Route\n");
  //  print(route);
  printf("Optimized Distance = %d\n", distance);
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
  //  printf("Initial route\n");
  //  int temp[] = {1, 8, 9, 19, 7, 6, 20, 18, 21, 16, 13, 12, 24, 15, 14, 22, 5, 25, 10, 11, 4, 2, 3, 23, 17};
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
  printf("\n");
}
