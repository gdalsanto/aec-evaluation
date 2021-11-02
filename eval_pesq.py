import soundfile as sf
import os
import sys 
import re
import inspect
import numpy as np
from pypesq import pesq as pesqMOS
from pesq import pesq as pesqMOSLQO
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from dataloader import get_file_line
# implementation of the P.862 standard https://github.com/vBaiCai/python-pesq



def compute_pesq_MOS(ref_list_path, deg_path):
    '''
    compute pesq score (ITU-T P.862)
    input 
        ref_list_path: list of reference wave files 
        deg_path: path to the folder with degradated files 
    '''

    ref_list = get_file_line(ref_list_path)    
    # deg_list = get_file_line(deg_path)
    deg_list = [os.path.join(deg_path,f) for f in os.listdir(deg_path) if os.path.splitext(f)[1]=='.wav']
    
    deg_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    ref_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    min_len = min(len(deg_list), len(ref_list))
    scores = []
    for idx, filename in enumerate(ref_list[:min_len]):
        ref, sr = sf.read(filename)
        deg, sr = sf.read(deg_list[idx])
        min_len = min(len(deg), len(ref))
        scores.append(pesqMOS(ref[:min_len], deg[:min_len], sr))
 
    print('average pesq (P.863): {}'.format(np.mean(scores)))

def compute_pesq_MOSLQO(ref_list_path, deg_path):
    '''
    compute pesq score (ITU-T P.862.2)
    input 
        ref_list_path: list of reference wave files 
        deg_path: path to the folder with degradated files 
    '''

    ref_list = get_file_line(ref_list_path)   
    deg_list = get_file_line(deg_path) 
    # deg_list = [os.path.join(deg_path,f) for f in os.listdir(deg_path) if os.path.splitext(f)[1]=='.wav']
    
    deg_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    ref_list.sort(key=lambda f: int(re.sub('\D', '', f)))

    scores = []
    for idx, filename in enumerate(ref_list):
        ref, sr = sf.read(filename)
        deg, sr = sf.read(deg_list[idx])
        min_len = min(len(deg), len(ref))
        scores.append(pesqMOSLQO(sr, ref[:min_len], deg[:min_len], 'wb'))
 
    print('average pesq (P.862.2): {}'.format(np.mean(scores)))


ref_list_path = 'evaluation/eval_list/sync_nearend_speech_test.lst'
deg_path = 'evaluation/eval_wav/synthetic_test_2021_10_15'
# deg_path = 'evaluation/eval_list/sync_nearend_mic_test.lst'
# compute_pesq_MOS(ref_list_path, deg_path)
compute_pesq_MOS(ref_list_path, deg_path)