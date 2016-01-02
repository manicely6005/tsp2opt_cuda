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
 * main.cpp
 *
 ******************************************************************************/

#include "algorithms.h"
#include <iostream>
#include <mpi.h>

int main(int argc, char * argv[])
{
  // Initialize MPI
  MPI::Init(argc, argv);
  
  // Get myid and # of processors 
  int numproc = MPI::COMM_WORLD.Get_size();
  int myid = MPI::COMM_WORLD.Get_rank();
  
  std::cout << "hello from " << myid << std::endl;
 
  // wait until all processors come here 
  MPI::COMM_WORLD.Barrier();
 
  if ( myid == 0 ) {
    // only myid = 0 do this
    std::cout << numproc << " processors said hello!" << std::endl;
  } else {
    std::cout << numproc << " processors said hello2!" << std::endl;
  }
  
  tsp test(argc, argv); //read in command line input
  test.two_opt();
  
  MPI::Finalize();
  exit(EXIT_SUCCESS);
}
