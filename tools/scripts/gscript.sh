for i in *.ent
do
echo 'set terminal png' > rysuj.gp
echo "set output '"$i".png'" >> rysuj.gp
echo "set size 1,1" >> rysuj.gp
echo "set origin 0,0" >> rysuj.gp
echo "set multiplot" >> rysuj.gp
echo "set size 0.5 ,1" >> rysuj.gp
echo "set origin 0,0" >> rysuj.gp
echo "set pm3d" >> rysuj.gp
echo "set hidden3d" >>rysuj.gp
echo "splot './"$i"'" >>rysuj.gp

echo "set size 0.5, 0.5" >> rysuj.gp
echo "set origin 0.5, 0.25" >> rysuj.gp
echo "unset pm3d" >> rysuj.gp

name=`basename $i ent`dat
echo $name

echo "plot './"$name"' w l ti''" >>rysuj.gp


gnuplot rysuj.gp

done

convert -delay 12  -quality 95 *.png movie.mpg