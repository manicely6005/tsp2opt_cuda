#include "algorithms.h"
#include <iostream>
#include <chrono>
#include <ctime>

int main(int argc, char * argv[])
{
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  
  tsp test(argc, argv); //read in command line input
  test.two_opt();
  
  end = std::chrono::system_clock::now();
  
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  
  std::cout << "finished computation at " << std::ctime(&end_time)
  << "elapsed time: " << elapsed_seconds.count() << "s\n\n";
  
  exit(EXIT_SUCCESS);
}
