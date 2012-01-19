#! /bin/bash
#for len in `seq 100 500 10000`
for len in `seq 10000 1000 100000`
do
    head -n $len ./examples/testcase.rea > ./tmp/tst_"$len".rea
#    for leni in {1..4}
#    do
#     scp -r /home/seba/codebase/ncm/tmp/tst_"$len".rea seba@192.168.44.14$leni:~/codebase/ncm/tmp
#    done
#    mpirun -n 33 --hostfile tmp/hostfile python base.py ./tmp/tst_"$len".rea 1 10
    mpirun -n 1 python base.py ./tmp/tst_"$len".rea 1 10
done
