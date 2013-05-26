#! /bin/bash
for i in *.ent
do
    echo $i
    python max_pos_find.py $i
done

