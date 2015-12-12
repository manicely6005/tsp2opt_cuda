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
 * edge_weight.h
 *
 ******************************************************************************/

#ifndef edge_weight_h
#define edge_weight_h

#include <stdio.h>
#include <vector>

class edge_weight
{
public:
  edge_weight(); // Constructor, takes filename to read from as input
  ~edge_weight();
  int euc2d(int *route, int num_cities, struct city_coords *coords); // Calculates the Euclidean distance to another city
  int geo(int *route, int num_cities, struct city_coords *coords); // Calculates the Geographical distance to another city
  int att(int *route, int num_cities, struct city_coords *coords); // Calculates the Geographical distance to another city
  int ceil2d(int *route, int num_cities, struct city_coords *coords); // Calculates the Geographical distance to another city
	
private:

};

#endif /* edge_weight_h */
