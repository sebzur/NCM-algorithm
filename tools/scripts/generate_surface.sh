#!/bin/bash

#zalozenie  pliki z sum korelacyjnych maja rozszerzenei mat ( od matrix)
for i in $1/*.mat
do
    echo $i
    python ../entropy_surface/surface_generator.py $i > $2/`basename $i mat`ent
done