#! /bin/bash
# $1 - katalof z danymi
# $2 - katalog do outputu
for i in $1/*.rea
do
    echo $i
    filename=${i##*/}
    echo $filename
    tail -n +2 $i > $2/"$filename"_tmp
    head -n $3 $2/"$filename"_tmp > $2/$filename
done

