import numpy as np
import librosa
import matplotlib.pyplot as plt

class NLMS():
    #initialize
    def __init__(self,mu,taps):
        self.mu=mu
        self.N=taps
        self.e=0
        self.w=np.zeros((taps,1))
        self.xd=np.zeros((taps,1))

    #updates the filter sample by sample and updates weights sample by sample
    def run(self,x,d):
        N=self.N
        #udpate delay line 
        self.xd[0]=x
        #compute filter output
        y = np.dot(self.w.T,self.xd)
        #compute input energy
        pw=np.linalg.norm(self.xd)
        pw=pw*pw
        #pw=pw*pw
        #calculate error
        e = d - y
        #normalize by input energy
        e= e/(pw+0.0000000001)
        #compute gradients
        Fw=self.mu*np.multiply(self.xd,e)
        
        Fw=Fw.reshape(Fw.shape[0],-1)
        #weight update
        self.w = self.w + Fw
    
        #filtered output post update
        y = np.dot(self.w.T,self.xd)
        #shift delay line
        self.xd[1:]=self.xd[:-1]
        self.e=e

        #return current output sample, current error sample and current weights
        return (y,e,self.w)



x= 0.5*librosa.tone(440,sr=8000,length=6000) + 0.4*librosa.tone(550,sr=8000,length=6000)
d=librosa.tone(550,sr=8000,length=6000)
#test wrapper for NLMS filter on aec data for single talk case
taps=256
lmscls= NLMS(0.1,taps)
w=np.zeros((taps,1))
y=np.zeros_like(x)
e=np.zeros_like(x)
for i in range(0,len(x)):
    (y[i],e[i],w2)=lmscls.run(x[i],d[i])

plt.figure()
ax1=plt.subplot(4,1,1)
ax1.plot(x)
ax1=plt.subplot(4,1,2)
ax1.plot(d)
ax1=plt.subplot(4,1,3)
ax1.plot(e)
ax1=plt.subplot(4,1,4)
ax1.plot(y)

plt.show()