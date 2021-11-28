#!/bin/bash
echo "Meas with malloc ************************"
nvprof ./excercise_2b.out 1000
nvprof ./excercise_2b.out 10000
nvprof ./excercise_2b.out 100000
nvprof ./excercise_2b.out 1000000
nvprof ./excercise_2b.out 10000000
nvprof ./excercise_2b.out 31600000
nvprof ./excercise_2b.out 100000000
nvprof ./excercise_2b.out 316000000
nvprof ./excercise_2b.out 1000000000
