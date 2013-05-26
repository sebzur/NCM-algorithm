#! /bin/bash
# $1 - katalof z danymi
# $2 - katalog do outputu
for i in $1/*.mat
do
    echo $i
#    tail -n +2 $i > tmp_rea_c
#    head -n 10000 tmp_rea_c > tmp.rea
    #mpirun -n 5 python ./NCM-algorithm/run.py $i --normalize --rsteps 1000 10 1 `basename $i dat`mat
    python cordim.py $i $2
done

