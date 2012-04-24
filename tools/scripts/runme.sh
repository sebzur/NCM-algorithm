#! /bin/bash
#bash lorenz_signal.sh
bash correlation_sum.sh $1 $2
bash generate_surface.sh $1 $2
#bash gscript.sh
#bash gscript_only_srfc.sh