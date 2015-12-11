//
//  edge_weight.h
//
//
//  Created by Nicely, Matthew A CIV (US) on 10/23/15.
//
//

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
