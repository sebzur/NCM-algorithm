from pylab import *
from numpy import *
from scipy.optimize import fmin, fmin_l_bfgs_b, fmin_tnc, fmin_slsqp

## Generating noisy data to fit
n = 30
xmin = 0.1
xmax = 5
x = linspace(xmin, xmax, n)
#y = fn(x) + rand(len(x))*0.2*(fn(x).max()-fn(x).min())

y = sqrt(x)

## Initial parameter value
#v0 = [(y[-1] - y[0])/(xmax-xmin), 0, xmin, xmax]
v0 = [xmin , xmax]

## Fitting

def error_function(parameters, x, y):
#    a = parameters[0]
#    b = parameters[1]

    x_min = parameters[0]
    x_max = parameters[1]

    indices, = nonzero(x >= x_min) and nonzero(x <= x_max)

    print indices, x_min, x_max
    if not len(indices) > 2:
        return 100
    x_comp = x[indices[0]:indices[-1]]

    y = y[indices[0]:indices[-1]]
    ar, br = polyfit(x_comp, y, 1)
    xr = polyval([ar,br], x_comp)
    #compute the mean square error
    err= sqrt(sum((xr-y)**2)/len(xr))
    v = err + (x_min ** 2) + (x_max - x[-1]) ** 2
    print indices, x_min, x_max, v

#    v = diff(y[indices[0]:indices[-1]]).mean()
#    v = (v** 2).sum()
    
#    print 'v=', v, abs((y[indices[0]:indices[-1]]/x_comp).mean())

#    print 'X', x
#    print 'Y', y

    return v
    
#    return ((a * x_comp + b - y[indices[0]:indices[-1]]) ** 2).sum()
    #return ((a * x_comp + b - y[indices[0]:indices[-1]]) ** 2).sum() + (x_min if x_min <=3 else 0) ** 2 + (x_max - x[-1] if x_max>= x[-1] else 0) ** 2

v = fmin(error_function, v0, args=(x,y), maxiter=10000, maxfun=100000, ftol=0.00000001)
# = fmin_slsqp(error_function, v0, args=(x,y), bounds=[(None, None), (None, None), (x[0], x[-1]),(x[0], x[-1])] )


## Plot
def plot_fit():
    print 'Estimater parameters: ', v
#    X = linspace(v[2], v[3],n*5)
    plot(x , y, 'ro')
    axvline(v[0], 0 , 2)
    axvline(v[1], 0 , 2)

plot_fit()
show()
