import numpy
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()

def ruelle_sum(signal, m, tau, r_range):
    """ Caclulates the ruelle sum for the signal data.
    """
    total = numpy.zeros(len(r_range))
    templates_count = len(signal) - (m -1) * tau
    factor = 1.0 /templates_count
    for i in range(rank, templates_count, size):
        template = numpy.array(signal[i:i + m:tau])
        local_total = numpy.zeros(len(r_range))
        for j in range(len(signal) - (m - 1) * tau):
            checked = numpy.array(signal[j:j+m:tau])
            local_total += (numpy.abs((template - checked)).max() <= r_range)
        total += numpy.log(local_total * factor)
    results = comm.gather(total, root=0)
    if rank == 0:
        summed = numpy.array(results).sum(axis=0)
        return summed * factor


def correlation_sum(signal, m, tau, r_range):
    """ Caclulates correlaction sum for a signal.

    For the provided embeding space, tau and treshold 
    calculates correlation sum for a signal entered and
    returns the number of counts

    """
    total = numpy.zeros(len(r_range))
    templates_count = len(signal) - (m -1) * tau
    ext_factor = 1.0 / templates_count
    int_factor = 1.0 / (templates_count - 1) # -1 due to self-matches exclusion
    for i in range(rank, templates_count, size):
        template = numpy.array(signal[i:i + m:tau])
        for j in range(len(signal) - (m - 1) * tau):
            if i == j:
                continue
            checked = numpy.array(signal[j:j+m:tau])
            total += (numpy.abs((template - checked)).max() <= r_range)

    results = comm.gather(total, root=0)
    if rank == 0:
        summed = numpy.array(results).sum(axis=0)
        return summed * ext_factor * int_factor


if __name__ == '__main__':

    data_file_name = sys.argv[1]
    m = int(sys.argv[2])
    tau = int(sys.argv[3])

    signal = numpy.loadtxt(data_file_name)

    r_std = signal.std()
    r_min =  r_std * 0.001
    r_max = r_std * 5.0
    r_step = (r_max - r_min)/100
    #print 'R_min', r_min, 'R_max', r_max
    r_range = numpy.arange(r_min, r_max, r_step)[::-1]

    #m_range = range(1, max_m + 1) # zawsze prawa strona nie wchodzi
    #for m in m_range[:1]:
    ra = ruelle_sum(signal, m, tau, r_range)
    if rank == 0:
        for i, v in enumerate(ra):
            print(r_range[i], v)

    
    
