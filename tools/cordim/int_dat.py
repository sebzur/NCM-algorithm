from numpy import *
from pylab import *
import sys
import scipy
import scipy.optimize
Y=load(sys.argv[1])
hold(True)
#this will hold the regression coefs and the corresponding ms
regressions=[]
regX=[]
X=Y[0,:][1:]
for i in range(1, len(Y)):
    #plotting the curves
    subplot(1,2,1)
    plot(X,Y[i,:][1:], '-k')
    t=X
    z=Y[i,:][1:]
    zN=[]
    tN=[]
    indeksy_down=find(z>-15.0)
    for ind in indeksy_down:
        zN.append(z[ind])
        tN.append(t[ind])
    zN=array(zN)
    indeksy_up=find(zN<-5)
    zOK=[]
    tOK=[]
    for ind in indeksy_up:
        zOK.append(zN[ind])
        tOK.append(tN[ind])
    if len(zOK)>1:
        cordim=polyfit(tOK,zOK,1)
        regressions.append(cordim[0])
        regX.append(Y[:,0][i])
m=array(regX)
dm=array(regressions)
error_min=lambda p, dm, m: dm-p[0]*(1-scipy.exp(-p[1]*m))
p=array([6.0,1.0])
p_result, success=scipy.optimize.leastsq(error_min, p.copy(), args=(dm, m))
print p_result

axis([0, X[-1], -20, 0.1])
subplot(1,2,2)
plot(m,dm, '+b')
plot(m, p_result[0]*(1-scipy.exp(-p_result[1]*m)), '-r')
text(0.4, 5, r"$x^{2}, \int_{-1}^{1}$", size=30)
show()
hold(False)
