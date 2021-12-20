echo "Starting with sweep of number of particles! \n"
srun -n 1 ./ex_bonus.out 1 2000 64 1
srun -n 1 ./ex_bonus.out 2 2000 64 1
srun -n 1 ./ex_bonus.out 5 2000 64 1
srun -n 1 ./ex_bonus.out 10 2000 64 1
srun -n 1 ./ex_bonus.out 21 2000 64 1
srun -n 1 ./ex_bonus.out 46 2000 64 1
srun -n 1 ./ex_bonus.out 100 2000 64 1
srun -n 1 ./ex_bonus.out 210 2000 64 1
srun -n 1 ./ex_bonus.out 460 2000 64 1
srun -n 1 ./ex_bonus.out 1000 2000 64 1
srun -n 1 ./ex_bonus.out 2100 2000 64 1
srun -n 1 ./ex_bonus.out 4600 2000 64 1
echo "Starting with sweep of batch size"
srun -n 1 ./ex_bonus.out 100 100000 16 0
srun -n 1 ./ex_bonus.out 100 100000 32 0
srun -n 1 ./ex_bonus.out 100 100000 64 0
srun -n 1 ./ex_bonus.out 100 100000 128 0
srun -n 1 ./ex_bonus.out 100 100000 256 0