# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:58:54 2022

@author: prsdk
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt


d,sr= librosa.load('../media/nearend_mic_fileid_0.wav')
x,sr=librosa.load('../media/farend_speech_fileid_0.wav')
echo,sr=librosa.load('../media/echo_fileid_0.wav')
near_end,sr= librosa.load('../media/nearend_speech_fileid_0.wav')

class AdaptiveFilters():
    def __init__(self,mu,taps):
        self.e=0
        self.w=np.zeros((taps,1))
        self.xd=np.zeros((taps,1))
        self.mu=mu
        self.N=taps
        
class PNLMS(AdaptiveFilters):
    #initialize
    def __init__(self,mu,taps,rho,delta):
        super().__init__(mu,taps)
        self.delta=delta
        self.rho=rho
        self.g=np.zeros_like(self.w)

    #updates the filter sample by sample and updates weights sample by sample
    def run(self,x,d):
        N=self.N
        #udpate delay line 
        self.xd[0]=x
        #compute filter output
        y = np.dot(self.w.T,self.xd)
        #compute input energy
        pw=np.linalg.norm(self.xd)
        #calculate error
        e = d - y
        #normalize by input energy
        e= e/(pw+0.0000001)

        #computing updates
        lk= max(np.abs(self.w))
        lk= max(self.delta,lk)
        for i,wk in enumerate(self.w):
            self.g[i]= max(abs(wk),self.rho*lk)

        gavg=np.mean(self.g)

        #compute gradients
        Fw=self.mu* np.multiply(self.xd,e)
        Fw= np.multiply(Fw,self.g/gavg)
        Fw=Fw.reshape(Fw.shape[0],-1)

        #weight update
        self.w = self.w + Fw
    
        #filtered output post update
        y = np.dot(self.w.T,self.xd)
        #shift delay line
        self.xd[1:N-1]=self.xd[0:N-2]
        self.e=e

        #return current output sample, current error sample and current weights
        return (y,e,self.w)
    
#test wrapper for NLMS filter on aec data for single talk case
taps=1024
lmscls= PNLMS(0.05,taps,0.1,0.1)
w=np.zeros((taps,1))
y=np.zeros_like(x)
e=np.zeros_like(x)
for i in range(0,len(x)):
    (y[i],e[i],w)=lmscls.run(x[i],echo[i])

plt.figure()
plt.plot(e)
plt.title("error in echo estimate")
plt.figure()
plt.plot(y)
plt.title("estimated echo")
plt.figure()
plt.plot(echo)
plt.title("actual mic captured echo during single talk (desired signal)")
plt.figure()
plt.plot(x)
plt.title("far end reference(input signal)")
