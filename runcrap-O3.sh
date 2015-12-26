#!/bin/bash

declare -a array=("burma14" "att48" "eil76" "bier127" "a280" "rd400" "d493" "att532" "d657" "d1655" "rl1889" "u2319" "d2103" "fl3795")

for i in "${array[@]}"
do
	./tsp_2opt "TSPLIB/${i}.tsp"
done
