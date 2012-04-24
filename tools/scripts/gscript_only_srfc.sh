for i in *.ent
do
echo 'set terminal png' > rysuj.gp
echo "set output '"$i".png'" >> rysuj.gp
echo "set pm3d" >> rysuj.gp
echo "set hidden3d" >>rysuj.gp
echo "splot './"$i"'" >>rysuj.gp

gnuplot rysuj.gp

done

#convert -delay 12  -quality 95 *.png movie.mpg