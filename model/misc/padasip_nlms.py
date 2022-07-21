import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import librosa
import librosa.display
from torch.utils.data import sampler
import torch.optim as optim
import json
import cv2
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from torchinfo import summary
from IPython.display import Audio
import IPython.display as ipd
import soundfile as sd
import padasip as dsp


d,sr= librosa.load('../dataset/AEC-Challenge/datasets/synthetic/nearend_mic_signal/nearend_mic_fileid_10.wav',sr=8000)
x,sr=librosa.load('../dataset/AEC-Challenge/datasets/synthetic/farend_speech/farend_speech_fileid_10.wav',sr=8000)
echo,sr=librosa.load('../dataset/AEC-Challenge/datasets/synthetic/echo_signal/echo_fileid_10.wav',sr=8000)
near_end,sr= librosa.load('../dataset/AEC-Challenge/datasets/synthetic/nearend_speech/nearend_speech_fileid_10.wav',sr=8000)

N=256
d1=d.reshape(d.shape[0],1)
x1=dsp.input_from_history(x,N)
L=len(x1)
d1=d1[:L]
h=dsp.filters.FilterNLMS(N,mu=0.1,w="zeros")

y,e,w=h.run(d1,x1)

plt.figure()
plt.plot(e)
plt.title("recovered near end")
plt.figure()
plt.plot(near_end)
plt.title("near end signal")
plt.figure()
plt.plot(y)
plt.title("estimated echo")
