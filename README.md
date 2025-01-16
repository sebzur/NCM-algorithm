# Norm Components Matrix (NCM) algorithm for the correlation sums calculation.

The correlation sums are the building block for the evaluation of correlation dimension 
and the sample entropy -- complexity parameters useful in the (medical) time series analysis.

This is the parallel implementation written in Python language.

To run this software you have to have MPI and mpi4py installed. The sample
bash command to run the programme would be:

$ `mpirun python run.py ncm_mpi /path/to/data/data_file.dat 10 1 /path/to/results/results_file.dat --normalize --rsteps 1000`



# The full usage info (the output from $ pyton run.py --help):


usage: `run.py [-h] [--normalize] [--rsteps RSTEPS] [--fmin FMIN] [--fmax FMAX]
              file m tau output`

NCM algorithm for correlation sums

positional arguments:
  - method           for calculating correlation sums (ncm_plain / ncm_mpi) 
  - file             Signal data - single column file or REA format
  - m                Embeding dimension (default: 10)
  - tau              Time lag (default: 1)
  - output           Output file name

optional arguments:
  - -h, --help       show this help message and exit
  - --normalize      Should the output be normalized?
  - --rsteps RSTEPS  Number of treshold values
  - --fmin FMIN      SD * f_min - lower value for tresholds range (default:
                   0.001)
  - --fmax FMAX      SD * f_max - upper value for tresholds range (default: 5.0)
  
  
  # Requirements
  
  - `sudo apt install libopenmpi-dev`
  - `pip install mpi4py`
  - `pip install numpy`
  
  
