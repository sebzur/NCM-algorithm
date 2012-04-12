#!/bin/bash

#zalozenie  pliki z sum korelacyjnych maja rozszerzenei mat ( od matrix)
for i in *.mat
do
echo $i
python surface_gen.py $i > `basename $i mat`ent
done