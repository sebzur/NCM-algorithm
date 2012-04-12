#! /bin/bash

# zakladam ze plii z sygnalem maja rozszerzenie .dat

for i in *.dat
do
echo $i
mpirun -n 2 python ./NCM-algorithm/run.py $i --normalize --rsteps 300 10 1 `basename $i dat`mat
done

