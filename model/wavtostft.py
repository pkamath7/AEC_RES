import os 
import numpy as np
import librosa 
import matplotlib.pyplot as plt
import cv2

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


def wavtoimg(filepath):
    fft_len=256
    x,sr=librosa.load(filepath)
    stft_x=librosa.stft(x, n_fft=fft_len, hop_length=fft_len//2, win_length=fft_len, window='hann', center=True, dtype=None, pad_mode='constant')


dir_path= "../dataset_daps/daps"

