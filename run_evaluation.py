import os, sys
import inspect
from datetime import datetime

# include parent folder in path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

ref_list = '\'evaluation/eval_list/sync_farend_speech_test.lst\''
des_list = '\'evaluation/eval_list/sync_nearend_mic_test.lst\''
aec_list = '\'evaluation/eval_list/sync_nearend_mic_aec_test.lst\''

out_folder = '\'evaluation/eval_wav/synthetic_test_' + datetime.today().strftime('%Y_%m_%d') + '.lst\''
model_path = '\'models_files/DCGRU_Net_22-200-46-train.pkl\''
device = '\'cpu\''
nsamples = 200

os.system( 'python3.7 evaluation.py' + ' --ref_list ' + ref_list + ' --des_list ' + des_list + ' --aec_list ' + aec_list + ' --out_folder ' + out_folder + ' --model_path ' + model_path + ' --device ' + device + ' --nsamples ' + str(nsamples))
