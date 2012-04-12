#! /bin/bash

# zrob Lorentza dla zmieniajacej sie sigmy od 1 do 40

i=1
while [ $(echo "$i < 40 "|bc) -eq 1  ]
do
i=`echo "$i+3" |bc`
python lor_script.py --sigma $i --ro 28 --beta 2.666 --output wynik
done
