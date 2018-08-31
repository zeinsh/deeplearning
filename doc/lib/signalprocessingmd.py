import math
import numpy as np
import pandas as pd

def getStatistics(X):
    N=X.shape[0]
    xmean=X.mean()
    ck=np.sum(X**2)/float(N)
    ck0=np.sqrt(ck)
    D=np.var(X)
    co=np.std(X)
    M3=np.sum((X-xmean)**3)/float(N)
    gama1=(M3)/np.power(co,3)
    M4=np.sum((X-xmean)**4)/float(N)
    gama2=(M4)/np.power(co,4)-3
    ret={'mean':xmean,
         'ck':ck,
         'ck0':ck0,
         'D':D,
         'co':co,
         'M3':M3,
         'gama1':gama1,
         'M4':M4,
         'gama2':gama2
        }
    return ret
# scale the domain
def getScaledDomain(dt,N):
    return [k*1.0/(dt*N) for k in xrange(N)]
def cutSignal(xk,N1,N):
    xknew=np.copy(xk)
    xknew[N1:,:]=0
    return xknew
# Get harmonic signal samples 
# A : amplitude
# f : signal frequency
# dt : 1/fsp duration between samples , fsp : sampling frequency
# N : number of samples
def harm(A,f,dt,N):
    pi=3.1416
    y=A*np.sin(np.arange(N)*2*pi*f*dt)
    desc="%dsin(2PI%.0ft)"%(A,f)
    y=y.reshape((N,1))
    return y,desc
# Get Signal with random spikes
# X : original signal [vector of numbers]
# num : number of random spikes
def spike(sig,num):
    X=sig+0
    n=X.shape[0]
    meanX=X.mean()
    value=np.max(abs(X))+10
    rnd=np.random.randint(n, size=num)
    dire=np.random.rand(num)
    for i in xrange(num):
        ind=rnd[i]
        if X[ind]>meanX or (X[ind]==meanX and dire[i]>0.5):
            X[ind]+=1.2*value
        else:
            X[ind]-=1.2*value
    return X
# Shift signal with const value
# X : original signal [vector of numbers]
# const: shifting const value
def shift(X,const):
    return X+const

#### Correlation
# auto correlation
def RxxStep(X,xmean,l):
    n=X.shape[0]
    cntX=X-xmean
    part1=cntX[:n-l]
    part2=cntX[l:n ]
    ret=np.multiply(part1,part2)
    return np.sum(ret)/float(n-l)
def Rxx(X,L):
    xmean=np.mean(X)
    ret=np.array([RxxStep(X,xmean,i) for i in xrange(L)])
    return ret/np.max(abs(ret))
def RxyStep(X,Y,l):
    n=X.shape[0]
    xmean=X.mean()
    ymean=Y.mean()
    cntX=X-xmean
    cntY=Y-ymean
    part1=cntX[:n-l]
    part2=cntY[l:n ]
    ret=np.multiply(part1,part2)
    return np.sum(ret)/float(n-l)
def Rxy(X,Y,L):
    xmean=np.mean(X)
    ymean=np.mean(Y)
    ret=np.array([RxyStep(X,Y,i) for i in xrange(L)])
    return ret/np.max(abs(ret))

####forier Transform
# get real part of forier transform
def getRen(xk,n,N):
    return sum([xk[k]*math.cos((2.0*np.pi*n*k)/N) for k in xrange(N)])/N
def getRe(xk,N):
    return np.array([[getRen(xk,n,N)] for n in xrange(N)])
# get Imaginary part of forier transform
def getImn(xk,n,N):
    return sum([xk[k]*math.sin((2.0*np.pi*n*k)/N) for k in xrange(N)])/N
def getIm(xk,N):
    return np.array([[getImn(xk,n,N)] for n in xrange(N)])
# calculate module of the forier serie 
def getCn(Re,Im,N):
    return [math.sqrt(Re[i]*Re[i]+Im[i]*Im[i]) for i in xrange(N)]
# get abs,module of fourier transform for visulalization
def fourierTransform_self(xk,N):
    im=getIm(xk,N)
    re=getRe(xk,N)
    return getCn(re,im,N)

def fourierTransform(xk):
    """ Fourier transform to be used instead of previously self written functions
	using numpy fast fourier transform function

	input: 
	  xk: signal of type numpy array with float values

        return:
	  fft: fast fourier transfrom for the signal
	  cn: module or absolute values for fft's values

    """
    fft=np.fft.fftn(xk)
    cn=np.sqrt(np.power(np.real(fft),2)+np.power(np.imag(fft),2))
    return fft,cn

#### Convolution
def getYk_conv(X,h,k):
    """
    get kth value of the convolution
    parameters:
       X : numpy array of shape (N,1)
       h : numpy array of shape (M,1)
       k : scalar integer
    returns:
       kth value of the convolution between X and h
    """
    N=X.shape[0]
    M=h.shape[0]
    num=min(k,M)
    ser=[X[k-l-1]*h[l] for l in xrange(num) if k-l-1<N]
    return sum(ser)
def convolution_self(X,h):
    """
    get kth value of the convolution
    parameters:
       X : numpy array of shape (N,1)
       h : numpy array of shape (M,1)
    returns:
       the convolution between X and h
    """
    N=X.shape[0]
    M=h.shape[0]
    return np.array([getYk_conv(X,h,k) for k in xrange(N+M)])
def convolution(a,v,mode='full'):
    """Returns the discrete, linear convolution of two one-dimensional sequences.

	Parameters
	----------
	a : (N,) array_like
	    First one-dimensional input array.
	v : (M,) array_like
	    Second one-dimensional input array.
	mode : {'full', 'valid', 'same'}, optional


	Returns
	-------
	out : ndarray
	    Discrete, linear convolution of `a` and `v`.	
    """
    assert len(a.shape)==1,"function accepts only one dimentional vector"
    assert len(v.shape)==1,"function accepts only one dimentional vector"
    return np.convolve(a,v,mode)
