# aec-evaluation
Module for the evaluation of Acoustic Echo Cancellation systems.  
## Evaluation metrics 
`aec_eval.py` assemble the implementation of the following metrics:
### Echo Return Loss Enhancement ([ERLE](#erle))
Objective metric that gives indication of additional signal loss applied by the echo canceller.  
ERLE is only appropriate when measured in a quiet room with no background noise and only for single talk scenarios. 
### Perceptual Evaluation of Speech Quality ([PESQ](#pesq))
Objective metric that models subjective tests commonly used in telecommunications.  
-**[P.862](#862)**: meant for narrowband only, and the output is represented as MOS score.  
-**[P.862.2](#8622)**: meant for both narrowband and wideband, and the output is represented as MOS-LQO score.  
### Short Time Objective Intelligibility measures ([STOI](#stoi))
Objective metric for the assessment of the intelligibility of noisy and enhanced sppech.  
## Python wrappers
The module `aec_eval.py` uses the following python wrappers/packages:
- P.862: [vBaiCai/python-pesq](https://github.com/vBaiCai/python-pesq).   
- P.862.2: [ludlows/python-pesq](https://github.com/ludlows/python-pesq).   
- STOI: [mpariente/pystoi](https://github.com/mpariente/pystoi).   
## Conda environment
The files needed to create the conda environment used for this branch are in `/env` directory.  
Using the specification text file `spec-file.txt`:
```
conda create --name torch_tf_speex --file spec-file.txt
```
Using the .yml file `aec-eval.yml`:
```
conda env create -f torch_tf_aec.yml
```
## References
<a name="erle"></a>
"ITU-T recommendation G.168: Digital network echo cancellers", Feb 2012.  
<a name="pesq"></a>
<a name="862"></a>
"ITU-T recommendation P.862: Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrow-band telephone networks ans speech codecs", Feb 2001.  
<a name="8622"></a>
"ITU-T recommendation P.862.2: Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs", Nov 2007.  
<a name="stoi"></a>
"C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech', ICASSP 2010, Texas, Dallas"
