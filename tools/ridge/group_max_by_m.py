import numpy
import glob

def process(f_names):
    data = {}
    for f_name in f_names:
        for m, line in enumerate(numpy.loadtxt(f_name)):
            if not data.has_key(m):
                data[m] = []
            data[m].append([f_name]+list(line[1:]))            
    
    for m in data:
        n_f_name = '%d_maxes' % (m + 1)
        d_file = open(n_f_name, 'w')
        for line in data[m]:
            d_file.write("%s\t%.20f\t%.20f\n" % (line[0], line[1], line[2]))
        d_file.close()
#        numpy.savetxt(n_f_name, data[m])

if __name__ == "__main__":
    process(glob.glob('*.max'))
