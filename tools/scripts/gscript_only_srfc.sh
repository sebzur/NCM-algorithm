for i in $1/*.ent
do
    echo 'set terminal png' > $2/rysuj.gp
    echo "set output '"$i".png'" >> $2/rysuj.gp
    echo "set pm3d" >> $2/rysuj.gp
    echo "set hidden3d" >> $2/rysuj.gp
    echo "splot '"$i"'" >> $2/rysuj.gp
    gnuplot $2/rysuj.gp
done

#convert -delay 12  -quality 95 *.png movie.mpg