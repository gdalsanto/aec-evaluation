import soundfile as sf
import matplotlib.pyplot as plt 
import os
import sys 
import re
import inspect
import numpy as np

# implementation of the P.862 standard https://github.com/vBaiCai/python-pesq
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from dataloader import get_file_line

sig_len = 2**17
frame_size = 512
def compute_erle(ref_list_path, est_path):
    '''
    input:
        ref_list: microphone signal
        est_list: estimated nearend speech
    '''
    ref_list = get_file_line(ref_list_path)    
    est_list = get_file_line(est_path)
    # est_list = [os.path.join(est_path,f) for f in os.listdir(est_path) if os.path.splitext(f)[1]=='.wav']
    
    # est_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    # ref_list.sort(key=lambda f: int(re.sub('\D', '', f)))    

    scores = []
    for idx, filename in enumerate(ref_list):
        ref, sr = sf.read(filename)
        est, sr = sf.read(est_list[idx])
        # min_len = min(len(est), len(ref))
        erle = []
        for i in range(sig_len):
            erle.append(10 * np.log10(np.mean(np.abs(ref[i:i + frame_size])**2) / np.mean(np.abs(est[i:i + frame_size])**2)))
        x = np.arange(0,sig_len)
        fig, axs = plt.subplots(3)
        axs[0].plot(x, erle)
        axs[0].set_title('ERLE')
        plt.xlim(0, sig_len)
        axs[1].specgram(ref[:sig_len],NFFT=frame_size*2,Fs=sr,noverlap=512,cmap='magma',scale='default')
        axs[1].set_title('Spectrogram farend')
        axs[2].specgram(est[:sig_len],NFFT=frame_size*2,Fs=sr,noverlap=512,cmap='magma',scale='default')
        axs[2].set_title('Spectrogram enhanced mic signal')
        plt.show()
    

ref_list_path = 'evaluation/eval_list/doubletalk_mic.lst'
est_path = 'evaluation/eval_list/doubletalk_DCGRU.lst'
compute_erle(ref_list_path, est_path)