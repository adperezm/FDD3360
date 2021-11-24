#!/bin/bash
echo "Meas with malloc ************************"
nvprof ./excercise_2a.out 1 1000
nvprof ./excercise_2a.out 1 10000
nvprof ./excercise_2a.out 1 100000
nvprof ./excercise_2a.out 1 1000000
nvprof ./excercise_2a.out 1 10000000
nvprof ./excercise_2a.out 1 31600000
nvprof ./excercise_2a.out 1 100000000
nvprof ./excercise_2a.out 1 316000000
nvprof ./excercise_2a.out 1 1000000000
echo "Meas with cuda malloc host **************"
nvprof ./excercise_2a.out 0 1000
nvprof ./excercise_2a.out 0 10000
nvprof ./excercise_2a.out 0 100000
nvprof ./excercise_2a.out 0 1000000
nvprof ./excercise_2a.out 0 10000000
nvprof ./excercise_2a.out 0 31600000
nvprof ./excercise_2a.out 0 100000000
nvprof ./excercise_2a.out 0 316000000
nvprof ./excercise_2a.out 0 1000000000
