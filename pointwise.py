import rea
import numpy
import sys

def correlation_sum(signal, m, tau, r):
    """ Caclulates correlaction sum for a signal.

    For the provided embeding space, tau and treshold 
    calculates correlation sum for a signal entered and
    returns the number of counts

    """
    total = 0
    factorA = len(signal) - (m - 1) * tau
    factorB = len(signal) - (m - 1) * tau - 1
    templates_count = len(signal) - (m -1) * tau
    #for i in range(len(signal) - (m -1) * tau):
    #for i in range(templates_count)[:int(0.1 * templates_count)]:
    for i in range(templates_count):
        template = numpy.array(signal[i:i + m:tau])
        for j in range(len(signal) - (m - 1) * tau):
            if i == j:
                continue
            checked = numpy.array(signal[j:j+m:tau])
            total += numpy.abs((template - checked)).max() <= r
    return total, 1.0/ (factorA * factorB), total * 1.0/ (factorA * factorB)

    
    
if __name__ == '__main__':

    data_file_name = sys.argv[1]
    m = int(sys.argv[2])
    tau = int(sys.argv[3])

    rea_reader = rea.REA(data_file_name, skiprows=0)
    signal = rea_reader.get_signal()
    print signal[:10]

    r_std = signal.std()
    r_min =  r_std * 0.001
    r_max = r_std * 5.0
    r_step = (r_max - r_min)/100
    #print 'R_min', r_min, 'R_max', r_max
    r_range = numpy.arange(r_min, r_max, r_step)[::-1]

    #m_range = range(1, max_m + 1) # zawsze prawa strona nie wchodzi
    #for m in m_range[:1]:
    for r in r_range:
        print r, '\t'.join(map(str, correlation_sum(signal, m, tau, r)))
