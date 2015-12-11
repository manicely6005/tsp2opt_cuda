//
//  edge_weight.cpp
//
//
//  Created by Nicely, Matthew A CIV (US) on 10/23/15.
//
//

#include "edge_weight.h"
#include "algorithms.h"
#include <cmath>

#define WIDTH 3

edge_weight::edge_weight()
{
}

//Destructor clears the deques
edge_weight::~edge_weight()
{
}

int edge_weight::euc2d(int *route, int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int j, xi, yi, xj, yj, xd, yd;
  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    // route[i] - 1 convert the 1 based arr to the 0 based coord
    xi = coords[route[i]-1].x;
    yi = coords[route[i]-1].y;
    xj = coords[route[j]-1].x;
    yj = coords[route[j]-1].y;
    
    xd = pow((xi - xj), 2.0);
    yd = pow((yi - yj), 2.0);
    
    distance += floor(sqrt(xd + yd) + 0.5);
  }
  return distance;
}

int edge_weight::ceil2d(int *route, int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int j, xi, yi, xj, yj, xd, yd;
  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    // arr[i] - 1 convert the 1 based arr to the 0 based coord
    xi = coords[route[i]-1].x;
    yi = coords[route[i]-1].y;
    xj = coords[route[j]-1].x;
    yj = coords[route[j]-1].y;
    
    xd = pow((xi - xj), 2.0);
    yd = pow((yi - yj), 2.0);
    
    distance += ceil(sqrt(xd + yd));
  }
  return distance;
}

int edge_weight::geo(int *route, int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int deg, j; 
  float xi, yi, xj, yj;
  double PI = 3.141492;
  double min, latitude_i, latitude_j, longitude_i, longitude_j, RRR, q1, q2, q3;
  
  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    // arr[i] - 1 convert the 1 based arr to the 0 based coord
    xi = coords[route[i]-1].x;
    yi = coords[route[i]-1].y;
    xj = coords[route[j]-1].x;
    yj = coords[route[j]-1].y;
    
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
  return distance;
}

int edge_weight::att(int *route, int num_cities, struct city_coords *coords)
{
  int distance = 0;
  int dij, tij, j, xi, xj, yi, yj, xd, yd;
  float rij;
  
  for (int i=0; i<num_cities; i++) {
    j = i + 1;
    
    // arr[i] - 1 convert the 1 based arr to the 0 based coord
    xi = coords[route[i]-1].x;
    yi = coords[route[i]-1].y;
    xj = coords[route[j]-1].x;
    yj = coords[route[j]-1].y;
    
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
  return distance;
}
