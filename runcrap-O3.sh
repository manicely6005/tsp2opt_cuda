#!/bin/bash

declare -a array=("eil76" "bier127" "a280" "rd400" "d493" "d657" "d1655" "rl1889" "u2319" "d2103" "fl3795")

for i in "${array[@]}"
do
	./tsp_cuda2opt "TSPLIB/${i}.tsp"
done
