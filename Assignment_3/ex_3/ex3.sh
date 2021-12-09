echo "2- Change in streams. N streams=1"
nvprof ./excercise_3.out 1 10000 10000 2000
echo "2- Change in streams. N streams=2"
nvprof ./excercise_3.out 2 10000 10000 2000
echo "2- Change in streams. N streams=4"
nvprof ./excercise_3.out 4 10000 10000 2000

echo "3- Changing batch size"
nvprof ./excercise_3.out 4 10000 100 1000
nvprof ./excercise_3.out 4 10000 1000 1000
nvprof ./excercise_3.out 4 10000 10000 1000
nvprof ./excercise_3.out 4 10000 100000 1000
nvprof ./excercise_3.out 4 10000 1000000 1000
nvprof ./excercise_3.out 4 10000 2000000 1000
