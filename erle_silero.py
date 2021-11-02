'''
Compute the Echo Return Loss Enhancement (ERLE) score 
Only the segments cointaining speech are taken 
To detect speech Silero Voice Acivity Detector is used 
Author: Gloria Dal Santo, Logitech - VC Audio Team

Plots ERLE score in dB scale
Plots ERLE vs Frequency
vsharma3@logitech.com
gdalsanto@logitech.com
'''


from matplotlib.pyplot import get
import torch
import os, sys, inspect
torch.set_num_threads(1)
from pprint import pprint
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from dataloader import get_file_line

def compute_erle(mic_list, enh_list):
    '''Compute Echo Return Loss Enhancement score 
    Input 
        mic_list: list of paths to the microphone signals 
        eng_list: list of paths to the enhanced microphone signals
    '''
    frame_size = 512
    # load pretrained Silero VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
    (get_speech_ts,
    _, _, read_audio,
    _, _, _) = utils
    mic_sig_list = get_file_line(mic_list)
    enh_sig_list = get_file_line(enh_list)
    # unpack file paths
    erle = []
    for indx, (mic, enh) in enumerate(zip(mic_sig_list, enh_sig_list)):
        mic_sig = np.float32(read_audio(mic))
        enh_sig = np.float32(read_audio(enh))
        # tail of the enh signal tends to be dropped druing processing
        # -> make them of the same length 
        min_len = min(len(mic_sig), len(enh_sig))
        mic_sig = mic_sig[:min_len]
        enh_sig = enh_sig[:min_len]
        # run VAD
        speech_timestamps = get_speech_ts(torch.tensor(mic_sig), model,
                                        num_steps=4)
        # create lists of start and end timestamps
        start = []
        end = []
        for idx, timestamp in enumerate(speech_timestamps):
            start.append(timestamp['start'])
            end.append(timestamp['end'])
        # compute ERLE on segment where voice is detected
        current_erle = []
        for start_idx, end_idx in zip(start, end):
            for n in range(start_idx, end_idx-frame_size):
                current_erle.append(10 * np.log10(np.mean(np.abs(mic_sig[n:n + frame_size])**2) / np.mean(np.abs(enh_sig[n:n + frame_size])**2)))
        try:
            erle.append(sum(current_erle)/len(current_erle))
        except:
            pass
    
    erle = np.array(erle)
    erle = np.delete(erle, np.argwhere(np.isnan(erle)))
    erle = np.delete(erle, np.argwhere(np.isinf(erle)))
    print(sum(erle)/len(erle))

mic_list = 'evaluation/eval_list/farend_singletalk_mic.lst'
enh_list = 'evaluation/eval_list/farend_singletalk_DCGRU_oct19.lst'
compute_erle(mic_list, enh_list)

# result with this setting: 23.842410363705238