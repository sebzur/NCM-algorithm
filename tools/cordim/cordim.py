from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
from scipy.optimize import leastsq
import os

filename = sys.argv[1]

data = loadtxt(filename)
matrix = log(data)

ax = subplot(121)
subplots_adjust(left=0.25, bottom=0.25)

r = matrix[:,0]

sd = data[0,0]/5.0
ent_index = []
for edge in [0.20, 0.15, 0.1]:
    for r_index in range(data.shape[0]):
        v = data[r_index,0]/sd
        print v, data[r_index,0], log(data[r_index,0]), data.shape[0], r_index
        if v < edge:
            log_cm_m = matrix[r_index, 1]
            log_cm_m_1 = matrix[r_index, 2]
            entropy = log_cm_m - log_cm_m_1
            ent_index.append([v, entropy])
            break

print 'Entropia', ent_index


MAIN = 15
c = matrix[:,MAIN]
c_2 = matrix[:,2]

l, = plot(r, c, lw=2, color='red')
l_2, = plot(r, c_2, lw=1, color='red')

m_range = arange(1, matrix.shape[1])

l_fit, = plot([0], [0], lw=1, color='blue')

ax2 = subplot(122)
slopes_plot, = plot(m_range, m_range, 'r.')
ax2.set_xlim([0, m_range[-1] + 1])
ax2.set_ylim([0, 10])


axcolor = 'lightgoldenrodyellow'
x_max_ax = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
x_min_ax = axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

x_min = Slider(x_min_ax, 'x min', r[-1], r[0], valinit=r[-1])
x_max = Slider(x_max_ax, 'x_max', r[-1], r[0], valinit=r[0], slidermin=x_min)


hline = ax2.axhline(0, 0, 30, color='black')

def update(val):
    min_val = x_min.val
    max_val = x_max.val

    filter = logical_and(r>min_val,r<max_val)

    new_r = r.compress(filter)

    slopes = []
    for column in m_range:
        cms = matrix[:, column]
        cms = cms.compress(filter)

        error_min=lambda p, y, x: y - (p[0] * x + p[1])
        fit=array([1, 0])
        fit, success = leastsq(error_min, fit.copy(), args=(cms, new_r))

        slopes.append(fit)

    l_fit.set_ydata(slopes[MAIN][0] * new_r + slopes[MAIN][1])
    l_fit.set_xdata(new_r)
    
    regressions = [x[0] for x in slopes]
    slopes_plot.set_ydata(regressions)

    m=m_range
    dm=array(regressions)
    error_min=lambda p, dm, m: dm-p[0]*(1-exp(-p[1]*m))
    p=array([6.0,1.0])
    p_result, success=leastsq(error_min, p.copy(), args=(dm, m))
    hline.set_ydata(p_result[0])
    draw()

x_min.on_changed(update)
x_max.on_changed(update)

resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Save', color=axcolor, hovercolor='0.975')
def reset(event):
    min_val = x_min.val
    max_val = x_max.val
    DM = hline.get_ydata()
    file = open(sys.argv[2], 'a')
    ff = filename.split('/')[-1]
    file.write("%s\t%.2f\t%.2f\t%.4f\t" % (ff, min_val, max_val, DM))
    file.write('\t'.join(map(lambda x: "%.2f\t%.2f" % (x[0], x[1]), ent_index)))
    file.write('\n')
    file.close()

button.on_clicked(reset)



show()

