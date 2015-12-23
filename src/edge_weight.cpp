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
 * edge_weight.cpp
 *
 * Multiple way to calculate distances between coordinates.
 *
 ******************************************************************************/

#include "edge_weight.h"
#include "algorithms.h"
#include <cmath>

edge_weight::edge_weight()
{
}

edge_weight::~edge_weight()
{
}

int edge_weight::euc2d(int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int j;
  float xi, yi, xj, yj, xd, yd;

  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    xi = coords[i].x;
    yi = coords[i].y;
    xj = coords[j].x;
    yj = coords[j].y;
    
    xd = pow((xi - xj), 2.0);
    yd = pow((yi - yj), 2.0);
    
    distance += (int) floor(sqrt(xd + yd) + 0.5);
  }
  return (distance);
}

int edge_weight::ceil2d(int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int j;
  float xi, yi, xj, yj, xd, yd;
  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    xi = coords[i].x;
    yi = coords[i].y;
    xj = coords[j].x;
    yj = coords[j].y;
    
    xd = pow((xi - xj), 2.0);
    yd = pow((yi - yj), 2.0);
    
    distance += (int) ceil(sqrt(xd + yd));
  }
  return (distance);
}

int edge_weight::geo(int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int deg, j; 
  float xi, yi, xj, yj;
  double PI = 3.141492;
  double min, latitude_i, latitude_j, longitude_i, longitude_j, RRR, q1, q2, q3;
  
  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    xi = coords[i].x;
    yi = coords[i].y;
    xj = coords[j].x;
    yj = coords[j].y;
    
    deg = (int) xi;
    min = xi - deg;
    latitude_i = PI * (deg + 5.0 * min/3.0) / 180.0;
    
    deg = (int) yi;
    min = yi - deg;
    longitude_i = PI * (deg + 5.0 * min/3.0) / 180.0;
    
    deg = (int) xj;
    min = xj - deg;
    latitude_j = PI * (deg + 5.0 * min/3.0) / 180.0;
    
    deg = (int) yj;
    min = yj - deg;
    longitude_j = PI * (deg + 5.0 * min/3.0) / 180.0;
    
    // The distance between two different nodes i and j in kilometers is then computed as follows:
    RRR = 6378.388;
    
    q1 = cos(longitude_i - longitude_j);
    q2 = cos(latitude_i - latitude_j);
    q3 = cos(latitude_i + latitude_j);
    
    distance += (int) (RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
  }
  return (distance);
}

int edge_weight::att(int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int dij, tij, j;
  float rij, xi, xj, yi, yj, xd, yd;
  
  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    xi = coords[i].x;
    yi = coords[i].y;
    xj = coords[j].x;
    yj = coords[j].y;
    
    xd = pow((xi - xj), 2.0);
    yd = pow((yi - yj), 2.0);
    
    rij = sqrt((xd + yd) / 10.0);
    
    tij = round(rij);
    
    if (tij < rij) {
      dij = tij + 1;
    } else {
      dij = tij;
    }
    distance += dij;
  }
  return (distance);
}
