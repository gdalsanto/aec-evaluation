'''
Compute the Echo Return Loss Enhancement (ERLE) score 
Author: Vikas Sharma, Logitech (Gaming Headsets) 
Adaptation to python: Gloria Dal Santo, Logitech - VC Audio Team

Plots ERLE score in dB scale
Plots ERLE vs Frequency
vsharma3@logitech.com
gdalsanto@logitech.com
'''
import os
import math 

import numpy as np
from numpy.core.numeric import Inf
from numpy.lib.npyio import save
import soundfile as sf
import matplotlib.pyplot as plt 
from scipy.signal import welch
from scipy.signal.windows import hann

import matplotlib as mpl

def compute_erle(mic_file, aec_file, save_plt=False, filename=None):
    '''
    Args:
        mic_file (string): path to microphone signal *.wav
        aec_file (string): path to acoustic echo canceller output signal *.wav
        save_plt (bool): if True the plots are saved
        filename (string): filename to be atted to the saved plots (without extension)
    '''
    output_directory = './output'
    plots_dir = os.path.join(output_directory, 'plots')
    if not os.path.isdir(output_directory):
        print("Creating output directory {}".format(output_directory))
        os.makedirs(output_directory, exist_ok=True)
    if save_plt:
        mpl.rcParams['lines.linewidth'] = 0.5

        if not os.path.isdir(plots_dir):
            print("Creating output directory {}".format(plots_dir))
            os.makedirs(plots_dir, exist_ok=True)        
        
    # read the .wav file 
    mic_sig, fs = sf.read(mic_file)
    aec_sig, fs = sf.read(aec_file)
    
    # check if one channel
    if len(mic_sig.shape) != 1:
        raise TypeError('Only single channel is supported')
    
    # compute input and output energies on a sample basis 

    #   make files of the same size
    min_len = min(mic_sig.shape[0], aec_sig.shape[0])
    mic_sig = mic_sig[:min_len]
    aec_sig = aec_sig[:min_len]
    #   initialize erle scores and energies 
    erle = np.zeros((min_len,))
    mic_eng = 0
    aec_eng = 0
    eps=1e-20
    smooth_val = 0.9999
    
    for indx, (mic_sample, aec_sample) in enumerate(zip(mic_sig, aec_sig)):
        mic_eng = smooth_val * mic_eng + (1 - smooth_val) * (mic_sample ** 2)
        aec_eng = smooth_val * aec_eng + (1 - smooth_val) * (aec_sample ** 2)
        erle[indx] = (mic_eng+eps)/(aec_eng+eps)
    
    
    # plot the result 
    fig1, ax1 = plt.subplots()
    time = np.divide(np.array(range(min_len)), fs)
    ax1.plot(time, 10*np.log10(erle+1e-20))
    ax1.set_title('Average ERLE score')
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('ERLE (dB)')
    ax1.grid(1)
    ax1.set_xlim(0, time[-1])
    ax1.set_ylim(-5, 100)
    if save_plt:
        filepath = os.path.join(plots_dir, 'erle_td_' + filename)
        try:
            fig1.savefig(filepath)
        except:
            print('if you want to save your plots then specify a filename')
        
    # frequency domain ERLE
    nfft = 512
    hop = 128
    # compute the periodograms
    f, Pyy = welch(mic_sig[math.floor(min_len/2):],
                   fs=fs,
                   window=hann(nfft),
                   noverlap=hop,
                   nfft=nfft,
                   return_onesided=True,
                   detrend=False)
    f, Pxx = welch(aec_sig[math.floor(min_len/2):],
                   fs=fs,
                   window=hann(nfft),
                   noverlap=hop,
                   nfft=nfft,
                   return_onesided=True,
                   detrend=False)
    erle_freq = 10*np.log10(Pyy) - 10*np.log10(Pxx)
    erle_freq = smooth_val * erle_freq + (1.0 - smooth_val) * erle_freq
    
    # plot the result 
    fig2, ax2 = plt.subplots()
    ax2.semilogx(f, erle_freq)
    ax2.set_title('ERLE score - periodogram')
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel('ERLE (dB)')
    ax2.grid(1)
    ax2.set_xlim(0, f[-1])
    if save_plt:
        filepath = os.path.join(plots_dir, 'erle_fd_' + filename)
        fig2.savefig(filepath)
                    
''' 
USAGE 
micData = '/Users/gloria/Documents/logitech/codes/misc/vikas/speexdsp_test_data/testOut/micsignal.wav'
sendOutData = '/Users/gloria/Documents/logitech/codes/misc/vikas/speexdsp_test_data/testOut/output_vikas_res.wav'
compute_erle(micData, sendOutData, save_plt=True, filename='plot_test')
'''    

