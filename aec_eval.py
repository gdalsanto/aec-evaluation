'''
Evaluation metrics for AEC systems 
'''

import torch
import os
import math 

import numpy as np
import soundfile as sf
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.signal.windows import hann

from pypesq import pesq as pesqMOS # ITU-T P.862
# https://github.com/vBaiCai/python-pesq
from pesq import pesq as pesqMOSLQO # ITU-T P.862.2
# https://github.com/ludlows/python-pesq 

from pystoi import stoi
# https://github.com/mpariente/pystoi

# ------------------------------ ERLE ------------------------------ #

def compute_erle(mic_file, aec_file, save_plt=False, filename=None):
    '''
    Computes the Echo Return Loss Enhancement
    
    Args:
        mic_file (string): path to microphone signal *.wav
        aec_file (string): path to acoustic echo canceller output signal *.wav
        save_plt (bool): if True the plots are saved
        filename (string): filename to be atted to the saved plots (without extension)
        
    Returns:
        erle_avg: average erle
        erle: sample based erle
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

    erle_avg = 10*np.log10(erle[np.nonzero(erle)]+eps)
    erle_avg = erle_avg.mean()
    
    # plot the result 
    fig1, ax1 = plt.subplots()
    time = np.divide(np.array(range(min_len)), fs)
    ax1.plot(time, 10*np.log10(erle+1e-20))
    ax1.set_title('Average ERLE score: {}'.format(erle_avg))
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('ERLE (dB)')
    ax1.grid(1)
    ax1.set_xlim(0, time[-1])
    ax1.set_ylim(-5, 100)
    if save_plt:
        filepath = os.path.join(plots_dir, 'erle_vikas_td_' + filename)
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
        filepath = os.path.join(plots_dir, 'erle_vikas_fd_' + filename)
        fig2.savefig(filepath)
        
    return erle_avg, erle
        
def compute_erle_silero(mic_file, aec_file, save_plt=False, filename=None):
    '''
    Compute Echo Return Loss Enhancement using Silero VAD to detect the segments of the input signals conatining speech
    Silero VAD: https://github.com/snakers4/silero-vad
    
    Args:
        mic_file (string): path to microphone signal *.wav (16kHz)
        aec_file (string): path to acoustic echo canceller output signal *.wav
        save_plt (bool): if True the plots are saved
        filename (string): filename to be atted to the saved plots (without extension)
    
    Returns:
        erle_avg: average erle
        erle: sample based erle
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

    # load pretrained Silero VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
    (get_speech_ts,
    _, _, read_audio,
    _, _, _) = utils

    # read the .wav file 
    mic_sig = np.float32(read_audio(mic_file))
    aec_sig = np.float32(read_audio(aec_file))
    fs = 16000
    
    # check if one channel
    if len(mic_sig.shape) != 1:
        raise TypeError('Only single channel is supported')

    # make files of the same size
    min_len = min(mic_sig.shape[0], aec_sig.shape[0])
    mic_sig = mic_sig[:min_len]
    aec_sig = aec_sig[:min_len]
    
    # run VAD
    speech_timestamps = get_speech_ts(torch.tensor(mic_sig), model, num_steps=4)
    # create VAD mask 

    mask = np.zeros((min_len, 1))
    for idx, timestamp in enumerate(speech_timestamps):
        split = range(timestamp['start'], timestamp['end'])
        mask[list(split)] = 1 
    
    # compute ERLE on segment where voice is detected and on a sample basis
    erle = np.zeros((min_len,))
    mic_eng = 0
    aec_eng = 0
    eps=1e-20
    smooth_val = 0.9999
    
    for indx, (mic_sample, aec_sample) in enumerate(zip(mic_sig, aec_sig)):
        mic_eng = smooth_val * mic_eng + (1 - smooth_val) * (mic_sample ** 2)
        aec_eng = smooth_val * aec_eng + (1 - smooth_val) * (aec_sample ** 2)
        erle[indx] = (mask[indx] * (mic_eng+eps))/(aec_eng+eps)

    erle_avg = 10*np.log10(erle[np.nonzero(erle)]+eps)
    erle_avg = erle_avg.mean()
    
    # plot the result 
    fig1, ax1 = plt.subplots()
    time = np.divide(np.array(range(min_len)), fs)
    ax1.plot(time, 10*np.log10(erle+1e-20))
    ax1.set_title('Average ERLE score {:.4f} dB'.format(erle_avg))
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('ERLE (dB)')
    ax1.grid(1)
    ax1.set_xlim(0, time[-1])
    ax1.set_ylim(-5, 100)
    if save_plt:
        filepath = os.path.join(plots_dir, 'erle_silero_td_' + filename)
        try:
            fig1.savefig(filepath)
        except:
            print('if you want to save your plots then specify a filename')
    
    return erle_avg, erle 
            
# ------------------------------ PESQ ------------------------------ #

def compute_pesq_MOS(ref_file, deg_file, print_score=False):
    '''
    Returns the pesq score (ITU-T P.862)
    
    input 
        ref_file: filepath to reference wave file
        deg_file: path to the folder with degradated files 
        print_score (bool): if True print the PESQ score
    '''
    # read the .wav file 
    ref_sig, fs = sf.read(ref_file)
    deg_sig, fs = sf.read(deg_file)
    
    if (len(ref_sig.shape) != 1) or (len(deg_sig.shape) != 1):
        raise TypeError('Only single channel is supported')

    # make files of the same size
    min_len = min(ref_sig.shape[0], deg_sig.shape[0])
    ref_sig = ref_sig[:min_len]
    deg_sig = deg_sig[:min_len]
    
    score = pesqMOS(ref_sig, deg_sig, fs)
    if print_score:
        print('PESQ (P.862) score: {:.4f}'.format(score))
        
    return score

def compute_pesq_MOSLQO(ref_file, deg_file, print_score=False):
    '''
    Returns the pesq score (ITU-T P.862.2)
    
    input 
        ref_file: filepath to reference wave file
        deg_file: path to the folder with degradated files 
        print_score (bool): if True print the PESQ score
    '''
    # read the .wav file 
    ref_sig, fs = sf.read(ref_file)
    deg_sig, fs = sf.read(deg_file)
    
    if (len(ref_sig.shape) != 1) or (len(deg_sig.shape) != 1):
        raise TypeError('Only single channel is supported')

    # make files of the same size
    min_len = min(ref_sig.shape[0], deg_sig.shape[0])
    ref_sig = ref_sig[:min_len]
    deg_sig = deg_sig[:min_len]
    
    score = pesqMOSLQO(fs, ref_sig, deg_sig, 'wb')
    if print_score:
        print('PESQ (P.862.2) score: {:.4f}'.format(score))
        
    return score

# ------------------------------ STOI ------------------------------ #

def compute_stoi(ref_file, deg_file, print_score=False):
    '''
    Returns the stoi score
    
    input 
        ref_file: filepath to reference wave file
        deg_file: path to the folder with degradated files 
        print_score (bool): if True print the PESQ score
    '''
    # read the .wav file 
    ref_sig, fs = sf.read(ref_file)
    deg_sig, fs = sf.read(deg_file)
    
    if (len(ref_sig.shape) != 1) or (len(deg_sig.shape) != 1):
        raise TypeError('Only single channel is supported')

    # make files of the same size
    min_len = min(ref_sig.shape[0], deg_sig.shape[0])
    ref_sig = ref_sig[:min_len]
    deg_sig = deg_sig[:min_len]
    
    score = stoi(ref_sig, deg_sig, fs, extended=False)
    if print_score:
        print('STOI score: {:.4f}'.format(score))
        
    return score