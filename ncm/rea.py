import numpy

class REA(object):
    """ Represents a REA time series. """

    def __init__(self, filename, skiprows=1, usecols=(1,), only_valid=True):
        """ Initializes the REA objects: a filename provided
        is opened and scanned for the RR data.

        If only valid is set, the data with the valid flag in REA file
        will be loaded.

        """
        # The standard REA files is builded up with three colums:
        # time
        # RR interval value [ms]
        # validity flag
        # we're loading only the middle one, knowing in advance, that this
        # is a small integer number.        
        #self.signal = numpy.loadtxt(filename, skiprows=skiprows, usecols=usecols, dtype=numpy.int16)
        self.signal = numpy.loadtxt(filename, skiprows=skiprows, usecols=usecols)
        #sh = numpy.random.shuffle(numpy.arange(self.signal.size))[:self.signal.size/2]
        if only_valid:
            # If only valid flag is set, we filter out all invalid RRs
            flag = numpy.loadtxt(filename, skiprows=skiprows, usecols=(3,), dtype=numpy.bool)
            #flg = map(lambda x: not bool(x), flag)
            self.signal = numpy.compress(flag, self.signal, axis=0)

    def get_signal(self):
        return self.signal
