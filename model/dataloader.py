from re import S
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchaudio.__version__)


RESAMPLE_RATE=8000



class DAPSDatasetHelper():

    #Get the dataset dictionary 
    def get_file_descriptors(self,dirpath):
        directory={}
        dataset_path=self.dir
        cwd= os.getcwd()
        for i , (dirpath, dirname, filename) in enumerate(os.walk(dataset_path)):
            if(dirpath!=dataset_path):
                dirname=dirpath.split("/")[-1]
                files={}
                file_list=[]
                index=0
                for file in filename:
                    filepath = os.path.join( dirpath, file)
                    if ( (filepath.endswith('.wav'))):
                        if(file.startswith('.')):
                            pass
                        else:
                            file_list.append(filepath)
                file_list.sort()
                if(len(file_list)>0):
                    for filepath in file_list:
                        files[index]=filepath
                        index+=1
                    directory[dirname]=files
        return directory

    #initialization 
    def __init__(self):
        self.sample_rate=8000
        self.dir= "./dataset_daps/daps"
        self.dataset_dict=self.get_file_descriptors(self.dir)

        #stft config
        #frame size in ms
        self.framesize=25
        self.fft_len=self.sample_rate*self.framesize//1000
        self.window_size=self.fft_len
        self.hop_len=self.fft_len//2
        self.num_files_per_category=len(self.dataset_dict["produced"].keys())

        indx=2
        self.keys={}
        for key  in self.dataset_dict.keys():
            if(key=="produced"):
                self.keys[1]=key
            else:
                self.keys[indx]=key
                indx+=1

    #get the indexed file and sample rate
    def get_indxd_file(self,indx,isLabel=False):
        if(isLabel):
            category=self.keys[1]
        else:
            category=self.keys[np.random.randint(2,len(self.keys))]
        data,sr= librosa.load(self.dataset_dict[category][indx])
        Id= self.dataset_dict[category][indx].split("/")[-1].split('.')[0]
        return (data,sr, Id)

    def resample_audio(self,file,sr):
        out = librosa.resample(file, orig_sr=sr, target_sr=self.sample_rate)
        return out

    #get the train data and label at given index 
    def get_data(self,indx):
        data,sr,Id_data = self.get_indxd_file(indx)
        label,sr,Id_label = self.get_indxd_file(indx,True)
        if(sr == self.sample_rate ):
            pass
        else:
            data= self.resample_audio(data,sr)
            label= self.resample_audio(label,sr)

        return (data,label,Id_data,Id_label)


    #get stft frames with 50% overlap
    def getFeatures(self,file):

        n_fft = self.fft_len
        win_length = self.window_size
        hop_length = self.hop_len

        # define transformation
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        # Perform transformation
        waveform=torch.from_numpy(file)
        spec = spectrogram(waveform)

        return spec
 


class DAPS(Dataset):
    def __init__(self):
        super().__init__(self)
        self.daps= DAPSDatasetHelper()

    def __getitem__(self, index):
        data,label,_,_=self.daps.get_data(index)
        data_spec=self.daps.getFeatures(data)
        label_spec=self.daps.getFeatures(label)
        return (data_spec,label_spec)

    def __len__(self):
        return (len(self.daps.keys)-1)*self.daps.num_files_per_category




#test  
d=DAPSDatasetHelper()
x,y,idx,idy=d.get_data(1)
spec_x= d.getFeatures(x)
sepc_x=spec_x.numpy()
#plt.figure()
#imgplot = plt.imshow(spec_x)
#print(sepc_x .shape)
Spec=librosa.stft(y, hop_length=64)
print(Spec .shape)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(Spec,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()