from pylab import plot, show
from scipy import arange, transpose, matrix, where
from scipy import log
from numpy import ndarray, array, std

def logistic_map(x, A, length):
    wynik=[]
    for i in range(length):
        x=A*x*(1-x)
        wynik.append(x)
    return wynik[200:]

def heaviside(x1,x2,r):
    if abs(x1-x2)<r: return 1 
    else: return 0

def heaviside2(x1,x2,r):
    if max(abs(x1-x2))<r: return 1 
    else: return 0

N=1000
print log(N)
dane=logistic_map(0.4, 3.99,N)
dane=array([1.2, 1.3, 1.1, 1.2, 1.3])
corsum_all=[]
corsum2_all=[]
R=arange(std(dane)/10, std(dane), std(dane)/10)
print 'WYNIK R:', R
plik=file("dane.csv", 'w')
for i in dane:
    plik.write(str(i)+'\n')
plik.close()

for r in R :
    corsum2=0
    for i,x in enumerate(dane):
        for k,y in enumerate(dane):
            if k!=10000000000 and k<len(dane)-1 and i<len(dane)-1:
                corsum2+=heaviside2(array([dane[i],dane[i+1]]),array([dane[k],dane[k+1]]),r)
    corsum2_all.append(corsum2*1/((N-200.0)*(N-200.0-1.0)))

for r in R :
    corsum=0
    for i,x in enumerate(dane):
        for k,y in enumerate(dane):
            if k!=100000000000:
                corsum+=heaviside(x,y,r)
    corsum_all.append(corsum*1/((N-200.0)*(N-200.0-1)))
#corsum_all=ndarray(corsum_all)
plot(log(R), log(corsum_all), '-.', log(R), log(corsum2_all), '-')
print(2*(log(array(corsum_all))-log(array(corsum2_all))))
show()
