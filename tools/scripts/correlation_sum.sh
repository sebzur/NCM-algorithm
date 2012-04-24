#! /bin/bash
N=5
m=10
t=1
steps=1000
# $1 - katalof z danymi
# $2 - katalog do outputu
for i in $1/*.rea
do
    echo $i
#    tail -n +2 $i > tmp_rea_c
#    head -n 10000 tmp_rea_c > tmp.rea
    #mpirun -n 5 python ./NCM-algorithm/run.py $i --normalize --rsteps 1000 10 1 `basename $i dat`mat
    mpirun -n $N python ../../run.py $i $m $t $2/`basename $i rea`mat --rsteps $steps --normalize
done

