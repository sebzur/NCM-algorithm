# SD=75.17, log(SD*0.1) =  2.0171671221953318, log(SD*0.2) = 2.7103143027552772
set style arrow 1 heads size screen 0.008,90 ls 2
set arrow from 2.017, -9 to 2.017, 0 as 1
set arrow from 2.71, -9 to 2.71, 0  as 1

set label 1 at 1.7, -1
set label 1 "0.1 * SD" tc lt 3
set label 2 at 2.73, -1
set label 2 "0.2 * SD" tc lt 3

set xlabel "Log(r)"
set ylabel "Log(Cm(r))"

p 'wynik_10k' u (log($1)):(log($2)) w lp pt 4 t 'm=1', 'wynik_10k' u (log($1)):(log($3)) w l t 'm=2', 'wynik_10k' u (log($1)):(log($4)) w l t 'm=3', 'wynik_10k' u (log($1)):(log($5)) w l t 'm=4', 'wynik_10k' u (log($1)):(log($6)) w l t 'm=5', 'wynik_10k' u (log($1)):(log($7)) w l t 'm=6', 'wynik_10k' u (log($1)):(log($8)) w l t 'm=7', 'wynik_10k' u (log($1)):(log($9)) w l t 'm=8', 'wynik_10k' u (log($1)):(log($10))  w l t 'm=9', 'wynik_10k' u (log($1)):(log($11))  w l t 'm=10'
pause -1