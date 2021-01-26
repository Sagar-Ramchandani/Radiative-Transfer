import numpy as np
import re,os
from matplotlib import pyplot as plt

#Constants
PLANCK=6.626e-34
BOLTZMANN=1.38e-23
C=3e8
dustDensity=1e-17

def readKappaData(fileName):
    THIS_FOLDER=os.path.dirname(os.path.abspath(__file__))
    fileName=os.path.join(THIS_FOLDER,fileName)
    print(fileName)
    f=list(open(fileName))[1::]
    f=list(map(lambda x: re.split(r'\s+',x)[:-1], [x for x in f]))
    nu=[]
    knu=[]
    for n,k in f:
        nu.append(float(n))
        knu.append(float(k))
    return (nu,knu)
    
def specificIntensity(planck,boltzmann,temp,freq,lightSpeed):
    return (2*planck*freq**3/lightSpeed)/(np.e**(planck*freq/(boltzmann*temp))-1)

def randomLogSpace(start,end,N):
    #Base is 10
    samples=np.random.uniform(low=np.log10(start),high=np.log10(end),size=(N))
    samples=10**samples
    '''We sort in order to plot them properly,
    and to also be able to use linear interpolation more 
    efficiently otherwise we would have to use a binary search 
    for every single value of the frequency'''
    samples.sort()
    return samples

def linearInterpolation(x,x0,y0,x1,y1):
    result=(y0+(x-x0)*(y1-y0)/(x1-x0))
    return result

def RungeKuttaOrder4(f,g,yi,xi,h):
    k1=-f(xi)*yi+g(xi)
    k2=-f(xi+h/2)*(yi+h*k1/2)+g(xi+h/2)
    k3=-f(xi+h/2)*(yi+h*k2/2)+g(xi+h/2)
    k4=-f(xi+h)*(yi+h*k3)+g(xi+h)
    y=yi+(h/6)*(k1+2*k2+2*k3+k4)
    return y

nu,knu=readKappaData("dustkappa_silicate_SIinp.sec")
knu=[y for x,y in sorted(zip(nu,knu))]
nu.sort()

freqSpace=randomLogSpace(1e13,2*1e15,101)
Iv=specificIntensity(PLANCK,BOLTZMANN,5777,freqSpace,C)

jv=np.zeros(len(freqSpace))
kv=[]
pos=0
vprev=[]
vnext=[]
prevExists=False
for v in freqSpace:
    Running=True
    nextExists=False
    while Running:
        if v>nu[pos]:
            Prev=pos
            prevExists=True
        if v<nu[pos]:
            Next=pos
            nextExists=True
        if prevExists and nextExists:
            res=linearInterpolation(v,nu[Prev],knu[Prev],nu[Next],knu[Next])
            vprev.append(nu[Prev])
            vnext.append(nu[Next])
            print(Next,Prev)
            kv.append(res)
            Running=False
        pos+=1

'''Testing code to see if linear interpolation
of the random points gives similar plot as original
plt.plot(freqSpace,kv)
plt.plot(nu,knu)
plt.show()'''

