#%% GENERAL SET-UP
import os
import numpy as np
from numpy import vstack
import mne
import io 
from PIL import Image, ImageDraw, ImageFont
import os.path
import pickle
import copy
import subprocess
import sys
import time
from glob import glob
import fnmatch
import scipy.stats as st
from scipy.stats import norm
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from functools import partial
import functools
from itertools import compress
from mne.preprocessing import annotate_muscle_zscore
from mne.preprocessing import ICA
from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel, Vectorizer, GeneralizingEstimator
import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec 
from statsmodels.stats.anova import AnovaRM
from statannotations.Annotator import Annotator
from scipy.spatial import cKDTree
import xarray as xr
import json
from pathlib import Path
try:
    import seaborn
except:
    print('Could not load Seaborn')
import pandas as pd
try:
    from PyTrack.formatBridge import generateCompatibleFormat # type: ignore
except:
    print('Could not load PyTrack')
try:
    import pingouin as pg  # for effect sizes and confidence intervals
except:
    print('Could not load Pingouin')
#%% LOCAL FUNCTIONS
# Saving multiple plots function
# https://www.tutorialspoint.com/saving-all-the-open-matplotlib-figures-in-one-file-at-once
def save_multi_image(filename):
    if script_location == 'BlueBear':
        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            try:
                fig.savefig(pp, format='pdf', dpi=500)
            except:
                plt.savefig(pp, format='pdf', dpi=500)
        pp.close()
    else:
        print("Cannot save PDF from the current script location")
# Eyelink file converter
# https://github.com/Leandil/EDF-to-ASC---EYELINK-DATA-CONVERTER/blob/master/checkasc.py
def checkasc(exe_path,edf_pathfolder):     
    cmd = os.path.join(exe_path,"edf2asc.exe") ## path and call of edf2asc.exe 
    os.chdir(edf_pathfolder) ## folder + file path - type str
    dir1 = edf_pathfolder ## file path - type str
    for root,dirs,files in os.walk(dir1):
        print(files)
        for file in sorted(files):
           print(file)
           if file.endswith(".edf"): ## check if edf file exists in the folder 
                   ascfile = file[:-3]+"asc" ## create ascfile name from the edf name
                   if not os.path.isfile(ascfile) : ## check if an ascfile with this name already exists
                       subprocess.run([cmd, os.path.join(dir1,file)]) ## if not, execute and convert edftoasc
# Converting a figure to an image
# https://www.geeksforgeeks.org/saving-a-plot-as-an-image-in-python/
def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    return img 
# creating adajceny with a gaussian kernel instead of the standard linear adjacency
def gaussian_adjacency_binary(n, sigma, threshold=0.01):
    """Create a Gaussian-based adjacency matrix and binarize it."""
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            adjacency[i, j] = np.exp(-((i - j) ** 2) / (2 * sigma ** 2))
    
    # Binarize the adjacency matrix
    adjacency[adjacency >= threshold] = 1
    adjacency[adjacency < threshold] = 0

    # Convert to sparse format for mne compatibility
    return csr_matrix(adjacency)
# Function to find the prime index X where Prime{X}Name matches targetName
# and determine the screen side (left or right)
def find_prime_info(row):
    for i in range(1, 5):
        prime_name = row.get(f'prime{i}Name')
        if prime_name == row['targetName']:
            side = 1 if i % 2 == 1 else 2 # 1=left, 2=right
            return pd.Series({'targetPrimedAt': int(i), 'primeScreenSide': int(side)})
    return pd.Series({'targetPrimedAt': None, 'primeScreenSide': None})
# handle np.arrays (as we switch from pickle to json in a late stage)
def to_json_safe(obj):
    # NumPy arrays → lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # NumPy scalar types → Python scalars
    if isinstance(obj, np.generic):
        return obj.item()

    # Path objects → strings
    if isinstance(obj, Path):
        return str(obj)

    # slice objects → explicit JSON-friendly dict
    if isinstance(obj, slice):
        return {
            "__slice__": True,
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step
        }

    # tuples → lists (safe for JSON)
    if isinstance(obj, tuple):
        return list(obj)

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


#%% GENERAL VARIABLES
try: # 'BlueBear' or 'CHBH_local'
    script_location = 'BlueBear' 
    root = 'INSERT DIR'
    os.listdir(root) # just to throw an error if the location is not found
except:
    script_location = 'local' 
    root = 'INSERT DIR'
    os.listdir(root) # just to throw an error if the location is not found  
# File paths
paths = {
        'analysis': os.path.join(root,'Analysis'), # Location of the files needed for the analyses
        'inp_BEH':  os.path.join(root,'BEH_Data'), # Location of the input data, the raw MEG file
        'outp_BEH': os.path.join(root,'Results','BEH'), # Location of the output data, the results related to the MEG file
        'inp_MEG':  os.path.join(root,'MEG_Data'), # Location of the input data, the raw MEG file
        'outp_MEG': os.path.join(root,'Results','MEG'), # Location of the output data, the results related to the MEG file
        'inp_EYE':  os.path.join(root,'EYE_Data'), # Location of the input data, the raw MEG file
        'outp_EYE': os.path.join(root,'Results','EYE') # Location of the output data, the results related to the MEG file
        }
# Task specific variables (pf = parafoveal task, mm = memory task)
task = {
        'org_events':{'pf/task'     :240,   # Start of the task (ParaFov)
                    'pf/fix1'       :252,   # First fixation onset (start trial)
                    'pf/fix2'       :248,   # Intermediate fixation onset (during trial)
                    'pf/pfov/s1/p0' :1,     # First parafoveal screen, no prime
                    'pf/pfov/s1/p1' :129,   # First parafoveal screen, prime 1-back
                    'pf/pfov/s1/p2' :65,    # First parafoveal screen, prime 2-back
                    'pf/pfov/s2/p0' :2,     # Second parafoveal screen, no prime
                    'pf/pfov/s2/p1' :130,   # Second parafoveal screen, prime 1-back
                    'pf/pfov/s2/p2' :66,    # Second parafoveal screen, prime 2-back
                    'pf/fov/ani'    :192,   # Foveal stimulus onset, category animal
                    'pf/fov/clo'    :160,   # Foveal stimulus onset, category clothing
                    'pf/fov/foo'    :144,   # Foveal stimulus onset, category food
                    'pf/fov/pla'    :136,   # Foveal stimulus onset, category plants
                    'pf/fov/veh'    :132,   # Foveal stimulus onset, category vehicle
                    'mm/task'       :224,   # Start of the task (Memory retrieval)
                    'mm/fix'        :252,   # Fixation onset
                    'mm/stim/old'   :24,    # old item
                    'mm/stim/new'   :36,     # new item
                    'mm/resp/hit'   :128,   # hit
                    'mm/resp/cr'    :64,    # correct rejection
                    'mm/resp/miss'  :32,    # miss
                    'mm/resp/fa'    :16,    # false alarm
                    'mm/resp/gso'   :8,     # guess old
                    'mm/resp/gsn'   :4      # guess new
                    }, 
        'new_events':{'pf/task'     :240,   # Start of the task (ParaFov)
                    'pf/fix1'       :252,   # First fixation onset (start trial)
                    'pf/fix2'       :248,   # Intermediate fixation onset (during trial)
                    'pf/pfov/s1/p0' :1,     # First parafoveal screen, no prime
                    'pf/pfov/s1/p1' :129,   # First parafoveal screen, prime 1-back
                    'pf/pfov/s1/p2' :65,    # First parafoveal screen, prime 2-back
                    'pf/pfov/s2/p0' :2,     # Second parafoveal screen, no prime
                    'pf/pfov/s2/p1' :130,   # Second parafoveal screen, prime 1-back
                    'pf/pfov/s2/p2' :66,    # Second parafoveal screen, prime 2-back

                    'pf/fov/ani/p0' :1920,  # Foveal stimulus onset, category animal, no prime
                    'pf/fov/clo/p0' :1600,  # Foveal stimulus onset, category clothing, no prime
                    'pf/fov/foo/p0' :1440,  # Foveal stimulus onset, category food, no prime
                    'pf/fov/pla/p0' :1360,  # Foveal stimulus onset, category plants, no prime
                    'pf/fov/veh/p0' :1320,  # Foveal stimulus onset, category vehicle, no prime
                    'pf/fov/ani/p1/l' :19211,  # Foveal stimulus onset, category animal, prime 1-back, left
                    'pf/fov/clo/p1/l' :16011,  # Foveal stimulus onset, category clothing, prime 1-back, left
                    'pf/fov/foo/p1/l' :14411,  # Foveal stimulus onset, category food, prime 1-back, left
                    'pf/fov/pla/p1/l' :13611,  # Foveal stimulus onset, category plants, prime 1-back, left
                    'pf/fov/veh/p1/l' :13211,  # Foveal stimulus onset, category vehicle, prime 1-back, left
                    'pf/fov/ani/p2/l' :19221,  # Foveal stimulus onset, category animal, prime 2-back, left
                    'pf/fov/clo/p2/l' :16021,  # Foveal stimulus onset, category clothing, prime 2-back, left
                    'pf/fov/foo/p2/l' :14421,  # Foveal stimulus onset, category food, prime 2-back, left
                    'pf/fov/pla/p2/l' :13621,  # Foveal stimulus onset, category plants, prime 2-back, left
                    'pf/fov/veh/p2/l' :13221,  # Foveal stimulus onset, category vehicle, prime 2-back, left

                    'pf/fov/ani/p0' :1920,  # Foveal stimulus onset, category animal, no prime
                    'pf/fov/clo/p0' :1600,  # Foveal stimulus onset, category clothing, no prime
                    'pf/fov/foo/p0' :1440,  # Foveal stimulus onset, category food, no prime
                    'pf/fov/pla/p0' :1360,  # Foveal stimulus onset, category plants, no prime
                    'pf/fov/veh/p0' :1320,  # Foveal stimulus onset, category vehicle, no prime
                    'pf/fov/ani/p1/r' :19212,  # Foveal stimulus onset, category animal, prime 1-back, right
                    'pf/fov/clo/p1/r' :16012,  # Foveal stimulus onset, category clothing, prime 1-back, right
                    'pf/fov/foo/p1/r' :14412,  # Foveal stimulus onset, category food, prime 1-back, right
                    'pf/fov/pla/p1/r' :13612,  # Foveal stimulus onset, category plants, prime 1-back, right
                    'pf/fov/veh/p1/r' :13212,  # Foveal stimulus onset, category vehicle, prime 1-back, right
                    'pf/fov/ani/p2/r' :19222,  # Foveal stimulus onset, category animal, prime 2-back, right
                    'pf/fov/clo/p2/r' :16022,  # Foveal stimulus onset, category clothing, prime 2-back, right
                    'pf/fov/foo/p2/r' :14422,  # Foveal stimulus onset, category food, prime 2-back, right
                    'pf/fov/pla/p2/r' :13622,  # Foveal stimulus onset, category plants, prime 2-back, right
                    'pf/fov/veh/p2/r' :13222,  # Foveal stimulus onset, category vehicle, prime 2-back, right

                    'mm/task'       :224,   # Start of the task (Memory retrieval)
                    'mm/task'       :224,   # Start of the task (Memory retrieval)
                    'mm/task'       :224,   # Start of the task (Memory retrieval)
                    'mm/fix'        :252,   # Fixation onset
                    'mm/stim/old'   :24,    # old item
                    'mm/stim/new'   :36,    # new item
                    'mm/resp/hit'   :128,   # hit
                    'mm/resp/cr'    :64,    # correct rejection
                    'mm/resp/miss'  :32,    # miss
                    'mm/resp/fa'    :16,    # false alarm
                    'mm/resp/gso'   :8,     # guess old
                    'mm/resp/gsn'   :4
                    },  
           'cats'   :['ani',                # animals
                      'clo',                # clothing
                      'foo',                # food
                      'pla',                # plants
                      'veh'                 # vehicles
                    ],
           'primes' :['p0',                # no prime
                      'p1',                # prime 1 screen back
                      'p2'                 # prime 2 screens back
                    ],
            'primes_lat' :['p0',                # no prime
                      'p1l',                # prime 1 screen back, left
                      'p1r',                # prime 1 screen back, right
                      'p2l',                 # prime 2 screens back, left
                      'p2r'                 # prime 2 screens back, right
                    ],
            'primes_lat_eve' :['p0',                # no prime
                    'p1/l',                # prime 1 screen back, left
                    'p1/r',                # prime 1 screen back, right
                    'p2/l',                 # prime 2 screens back, left
                    'p2/r'                 # prime 2 screens back, right
                ],
           'pr_lab' :['no prime',
                      '1-back prime',
                      '2-back prime'
                    ],
            'pr_lab_lat' :['no prime',
                      '1-back prime left',
                      '1-back prime right',
                      '2-back prime left',
                      '2-back prime right'
                    ]
        }
# MEG specific variables
meg = {
       'samp_freq'          :1000,
       'chan_trg'           :'STI101',
       'chan_trg_eog'       :'STI_EOG',
       'chan_eog'           :['HEOG', 'VEOG'],
       'chan_veog'          :'VEOG',
       'chan_heog'          :'HEOG',
       'chan_bli'           :'EYE_bli',
       # use mne.pick_channels_regexp() for the chan_occ one
       'chan_occ'           :'MEG163.|MEG184.|MEG183.|MEG201.|'+ # Left posterior parietal
                             'MEG224.|MEG202.|MEG223.|MEG244.|'+ # Right posterior parietal
                             'MEG172.|MEG164.|MEG191.|MEG204.|'+ # Left occipital
                             'MEG194.|MEG171.|MEG173.|MEG192.|'+ # Left occipital
                             'MEG211.|MEG193.|MEG174.|MEG214.|'+ # Left occipital
                             'MEG203.|MEG231.|MEG243.|MEG232.|'+ # Right occipital
                             'MEG252.|MEG253.|MEG251.|MEG234.|'+ # Right occipital
                             'MEG212.|MEG233.|MEG254.|MEG213.'   # Right occipital
                             ,
       'chan_RS_g'         : ['MEG2041',
                            'MEG2031',
                            'MEG1921',
                            'MEG2111',
                            'MEG2341',
                            'MEG2011',
                            'MEG2021',
                            'MEG1911',
                            'MEG2311',
                            'MEG1941',
                            'MEG2321',
                            'MEG1931',
                            'MEG2331',
                            'MEG2121',
                            'MEG1731',
                            'MEG2511',
                            ], # gradiometer channels, labeled onto the magnetometer channels (for ERF)
        'chan_RS_gp'         : ['MEG2042',
                            'MEG2032',
                            'MEG1922',
                            'MEG2112',
                            'MEG2342',
                            'MEG2012',
                            'MEG2022',
                            'MEG1912',
                            'MEG2312',
                            'MEG1942',
                            'MEG2322',
                            'MEG1932',
                            'MEG2332',
                            'MEG2122',
                            'MEG1732',
                            'MEG2512',
                            'MEG2043',
                            'MEG2033',
                            'MEG1923',
                            'MEG2113',
                            'MEG2343',
                            'MEG2013',
                            'MEG2023',
                            'MEG1913',
                            'MEG2313',
                            'MEG1943',
                            'MEG2323',
                            'MEG1933',
                            'MEG2333',
                            'MEG2123',
                            'MEG1733',
                            'MEG2513',
                            ], # gradiometer pairs (for TFR)
        'chan_RS_ml'         :['MEG1941',
                            'MEG1921',
                            'MEG1731',
                            'MEG1931'], # magnetometers channels in the left hemisphere (for ERF and TFR)
        'chan_RS_mr'         :['MEG2321',
                            'MEG2341',
                            'MEG2511',
                            'MEG2331'], # magnetometers channels in the right hemisphere (for ERF and TFR)
        'chan_RE_g'         : ['MEG2041',
                            'MEG2031',
                            'MEG1921',
                            'MEG2111',
                            'MEG2341',
                            ], # gradiometer channels (for ERF)
        'chan_RE_gp'         : ['MEG2042',
                            'MEG2032',
                            'MEG1922',
                            'MEG2112',
                            'MEG2342',
                            'MEG2043',
                            'MEG2033',
                            'MEG1923',
                            'MEG2113',
                            'MEG2343',
                            ], # gradiometer pairs (for TFR)
        'chan_RE_ml'         :['MEG1511',
                                'MEG0241',
                                'MEG1521',
                                'MEG1611',
                                'MEG1721',
                                'MEG1641',
                                'MEG1731',
                                'MEG1941'], # magnetometers channels in the left hemisphere (for ERF and TFR)
        'chan_RE_mr'         :['MEG1331',
                                'MEG2611',
                                'MEG2421',
                                'MEG2641',
                                'MEG2431',
                                'MEG2521',
                                'MEG2321',
                                'MEG2511'], # magnetometers channels in the right hemisphere (for ERF and TFR) 
        'chan_gl'            : ['MEG1731', 'MEG1911', 'MEG1921', 'MEG1931', 'MEG1941', 'MEG2011', 'MEG2041'], # gradiometer channels (for ERF) 
        'chan_gr'            : ['MEG2021', 'MEG2031', 'MEG2311', 'MEG2321', 'MEG2331', 'MEG2341', 'MEG2511'], # gradiometer channels (for ERF)
        'chan_gpl'           : ['MEG1732', 'MEG1733', 'MEG1912', 'MEG1913', 'MEG1922', 'MEG1923', 'MEG1932', 'MEG1933', 'MEG1942', 'MEG1943', 'MEG2012', 'MEG2013', 'MEG2042', 'MEG2043'], # gradiometer pairs (for TFR)
        'chan_gpr'           : ['MEG2022', 'MEG2023', 'MEG2032', 'MEG2033', 'MEG2312', 'MEG2313', 'MEG2322', 'MEG2323', 'MEG2332', 'MEG2333', 'MEG2342', 'MEG2343', 'MEG2512', 'MEG2513'], # gradiometer pairs (for TFR)
        'chan_gpm'           : ['MEG2112', 'MEG2113', 'MEG2122', 'MEG2123'], # gradiometer pairs (for TFR)
        'chan_ecg'           : ['MEG1411','MEG1421','MEG1431',
                               'MEG1511','MEG1521','MEG1531',
                               'MEG2411','MEG2421','MEG2431',
                               'MEG2511','MEG2521','MEG2531',
                               ],
       #'chan_ecg'           : 'ECG003',
       'chan_all'           : 'meg',
       'chan_ias'           : ['IASX+',
                            'IASX-',
                            'IASY+',
                            'IASY-',
                            'IASZ+',
                            'IASZ-',
                            'IAS_DX',
                            'IAS_DY',
                            'IAS_X',
                            'IAS_Y',
                            'IAS_Z'],
       # https://www.chbh.bham.ac.uk/wiki/index.php/MaxFilter_Fine-Calibration_Files
       'crosstalk_file'     : 'ct_sparse_triux2.fif', # The crosstalk compensation file has parameters used to reduce interference between co-located magnetometer and paired gradiometer sensor units.
       'calibration_file'   : 'sss_cal_3140_60_190213.dat' # The calibration file contains information specfic to the local static magnetic fields and gradients.
       }
# Eyetracker specific variables
eye = {
       'device'         :'eyelink',         # hardware used
       'stim_list_mode' :'NA',              # not sure what this means
       'start_str'      :'EXP_START',       # message sent at the start of the experiment
       'stop_str'       :'EXP_END',         # message sent at the end of the experiment
       'eye'            :'L',               # eye recorded from
       'samp_freq'      :1000,              # sampling frequency
       'samp_freq_off'  :1-.00049369,       # it appears that the sampling frequency of the eyelink is slightly faster than that of MEG, this is the approximate difference
       'disp_w'         :1920,              # width of the display
       'disp_h'         :1080,              # height of the display
       'aoi'            :[0,0,1920,1080],   # area of interest, of not needed, use [0,0,eye['disp_w'],eye['disp_h']]
       'events'         :{'eog/bli'       :50,    # start blinks
                          'eog/fix'       :51,    # start fixation
                          'eog/sac'       :52,    # start saccade
                          'eog/bli_c'     :60,    # blinks continued
                          'eog/fix_c'     :61,    # fixation continued
                          'eog/sac_c'     :62     # fixation continued
                           },
       'subj_bad'       : {'110', '123', '132'}, # no or bad eyelink data
       'eog_adj'        : 10000000
       }
# Analysis specific variables
analysis = {
            'type'          :'SVM', # 'SVM' or 'ERFTFR'
            'types'         : {'erf'      : False, # ERP analysis
                               'tfr'      : False, # Time-frequency analysis
                               'mvpa'     : True, # multivariate pattern analysis
                               'ga'       : True, # grand average
                               'dc'       : False, # data checks
                               'behav'    : False # group level behavioral analyses
                               },
            'rej'           : dict(mag  = 5000e-15,     # 5000 fT    # based on FLUX
                                    grad = 5000e-13,    # 5000 fT/cm     # based on FLUX
                                    ),
             'bp_fil_ica'   : {'l_freq'     : 1,
                               'h_freq'     : 40,
                               'eog_l_freq' : 1, # all below frequencies based on the find_bads_eog/find_bads_ecg defaults
                               'eog_h_freq' : 10,
                               'ecg_l_freq' : 8,
                               'ecg_h_freq' : 16 
                               },
             'resmpl'       : 500,
             'erf'          : {'basewin'        : (-.1,0),
                               'cbp_times'      : (0, .500), # 0 - 500 ms 
                               'cbp_clalpha'    : 0.005, # cluster forming alpha that will determine cluster forming t-value threshold 
                               'cbp_nperm'      : 10000, # number of permutations
                               'vep_calc'       : False # whether to do the VEP calculation or not.
                               },
             'tfr'          : {'freqs'          : np.arange(30, 101, 2),
                              'n_cycles'        : np.arange(30, 101, 2) / 4 , # freqs / 4, the np.array should be the same as in the line above
                              'n_cycles_R2'     : np.arange(30, 101, 2) / 8 , # freqs / 8, the np.array should be the same as in the line above
                              'time_win'        : 4.0,

                              'gamma_freqs'     : np.arange(30, 81, 2), # 65-85 Hz (may want to look at 30-50 because of Peter paper)
                              'cbp_times'       : (0, .500), # 0 - 500 ms
                              'cbp_clalpha'     : 0.005, # cluster forming alpha that will determine cluster forming t-value threshold 
                              'cbp_nperm'       : 10000, # number of permutations 
                              },
             'mvpa'         : {'twin'           : .05, # average over a 50 ms time window, False if no time window is used 
                               'concat'         : True, # if this is true data is concatenated when windowed, not averaged
                               'temgen'         : False, # do to the temporal generalization or not
                               'cbp_times'      : (0, .500), # 0 - 250 ms 
                               'cbp_clalpha'    : 0.005, # cluster forming alpha that will determine cluster forming t-value threshold 
                               'cbp_nperm'      : 10000, # number of permutations
                               },
             'pltcol'       : ['chocolate','teal','indigo'],
             'pltcol_lat'   : ['chocolate','teal', 'green', 'indigo', 'purple'],
             'ica_chk'      : True,
             'ica_mnl'      : True,
             'times'        : {'prime2'     : (-.700, -.450), # in seconds relative to stim onset
                               'prime1'     : (-.350, -.100), # in seconds relative to stim onset
                               'target'     : (0, .500), # in seconds relative to stim onset
                               } 
            }

#%% SUBJECT VARIABLES
subj = {
        'nr'        :'',
        'bad'       :False
        }
# create a variable dict to store all the variables above
vars = {'paths'     :paths,
        'task'      :task,
        'meg'       :meg,
        'eye'       :eye,
        'analysis'  :analysis,
        }
# create a function for all the preprocessing
def MEG_preproc(vars,subj):
    # this function will perform all the preprocessing steps of the MEG analysis
    # input arguments:
    # vars = all the variable of the experiment in a dictionary
    #
    # get the variables out of the dictionary
    locals().update(vars)
    #%% SAVED VARIABLES LOCATIONS
    var_loc = {
                'eye' : os.path.join(paths["outp_EYE"],subj['nr']+'_eye_trg.csv'), # where the Eyetracker data with the triggers is stored
                'raw' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_raw.fif'), # where the (pretty much) raw data is saved
                'sss' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_raw_sss.fif'), # where the post SSS/Maxfilter data is saved
                'hpi' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_chpi.fif'), # where the post-cHPI data is saved
                'hps' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_headpos.obj'), # where the head position data is saved
                'fil' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_fil.fif'), # where the filtered data is saved
                'eve' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_events.fif'), # where the events are saved
                'bad' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_bads.fif'), # where the data with the bad sensors marked are saved
                'ano' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ano.fif'), # where the data with the annotations are saved
                'eog' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_eog.fif'), # where the post EOG channel (correction) data is saved
                'evb' : os.path.join(paths["outp_EYE"],analysis['type'],subj['nr']+'_eve_bli.npy'), # where the blink events are saved
                'evs' : os.path.join(paths["outp_EYE"],analysis['type'],subj['nr']+'_eve_sac.npy'), # where the blink events are saved
                'ica' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ica.fif'), # where the post ICA data is saved
                'pic' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_postica.fif'), # where the post ICA data is saved
                'epo' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_epo.fif'), # where the epoched data is saved
                'elk' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_elk.fif'), # where the epoched eyelink data is saved
                'sub' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json'), # where the subject data is saved
                }
    # set up the preprocessing choices based on the future analyses
    if analysis['type']=='ERFTFR':
        analysis['preproc'] = {'Max'    :True, # do the Maxfilter or not (ERF, TFR)
                               'cHPI'   :True, # do the cHPI correction or not (ERF, TFR)
                               'intrp'  :False} # to interpolate bad channels or not (SVM)
    elif analysis['type']=='SVM':
        analysis['preproc'] = {'Max'    :False, # do the Maxfilter or not (ERF, TFR)
                               'cHPI'   :False, # do the cHPI correction or not (ERF, TFR)
                               'intrp'  :True} # to interpolate bad channels or not (SVM)
    cur_redo = False # True for recalculate, False for load in old data
    if not glob(var_loc['epo']+'*') or cur_redo: # check if all preprocessing steps have been done already
        print('Subject:',subj['nr'])
        # save the subject data
        if not os.path.isfile(var_loc['sub']):
            with open(var_loc["sub"], "w", encoding="utf-8") as fp:
                json.dump(subj, fp, indent=2, default=to_json_safe)
        #%% PARTICIPANT EXCEPTIONS
        if subj['nr'] == '101':
            # no cHPI
            analysis['preproc']['cHPI'] = False
        if subj['nr'] in eye['subj_bad']:
            # no (good) eyelink data
            eye_df = pd.DataFrame({'StimulusName' : [],'Timestamp' : []})
        with open(var_loc["sub"], "r", encoding="utf-8") as fp:
            subj = json.load(fp)
        if not os.path.isfile(var_loc['eye']) and subj['nr'] not in eye['subj_bad']: # skip participant 110 since the eyelink recording went wrong there
            # EYELINK DATA
            # Read in the Eyelink data to get the triggers from it
            print('\nREAD EYELINK FILE\n')
            # https://github.com/Leandil/EDF-to-ASC---EYELINK-DATA-CONVERTER
            # convert the .EDF files to .ASC files
            # edf2asc.exe and edfapi.dll need to be in the analysis folder
            exe_path = paths['analysis']
            edf_pathfolder = paths['inp_EYE']
            checkasc(exe_path,edf_pathfolder)
            # https://github.com/titoghose/PyTrack
            # https://readthedocs.org/projects/pytrack-ntu/downloads/pdf/latest/
            # Convert data to generate csv file 
            if not os.path.isfile(os.path.abspath(os.path.join(paths['inp_EYE'],subj['nr']+'.csv'))):
                generateCompatibleFormat(exp_path       = os.path.abspath(os.path.join(paths['inp_EYE'],subj['nr']+'.asc')),
                                        device          = eye['device'],
                                        stim_list_mode  = eye['stim_list_mode'],
                                        start           = eye['start_str'],
                                        stop            = eye['stop_str'],
                                        eye             = eye['eye'])
            # read in that CVS file
            eye_df = pd.read_csv(os.path.abspath(os.path.join(paths['inp_EYE'],subj['nr']+'.csv')))
            # adjust for the difference in sampling frequency between Eyelink and MEG
            eye_df.Timestamp = eye_df.Timestamp+(eye_df.iloc[:,0] - eye_df.iloc[:,0]*eye['samp_freq_off'])
            eye_df.Timestamp = eye_df.Timestamp.astype(int)
            # lazy fix as I am tired, but if there are two of the samevalues for the Timestamps, this will give an error in the code, so the quick fix is to just delete one of them.
            if subj['nr'] == '120':
                eye_df = eye_df.drop(1713522)
            # start timer
            start_time = time.perf_counter()
            # adjust for the missing data due to the drift check and sampling rate
            while len(eye_df) != int(eye_df['Timestamp'].iat[len(eye_df.Timestamp)-1]) - int(eye_df['Timestamp'].iat[0]) +1:
                # find the moments where there is a jump in time
                idx = eye_df['Timestamp'][eye_df['Timestamp'].diff()>1].iat[0]
                txt = 'Progress: ' + str(round(eye_df[eye_df.Timestamp==idx].index[0]/len(eye_df)*100,2)) + '%'
                print('\r%s' % txt, end=' ')
                # Calculate elapsed time
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print (', Elapsed time: ', round(elapsed_time/60,2), 'minutes', end='')
                # get the beginning and end of the missing data
                end = eye_df[eye_df.Timestamp==idx]
                srt = eye_df.iloc[end.index[0]-1]
                # fill it in with the last values
                tmp = srt.copy()
                if int(end.Timestamp)-int(srt.Timestamp)>2:
                    for r in range((int(end.Timestamp)-int(srt.Timestamp+1))-1):
                        tmp = vstack([tmp, srt])
                    # convert it to a dataframe
                    tmp = pd.DataFrame(tmp, columns=eye_df.columns)
                    tmp.Timestamp = list(range(int(srt.Timestamp+1),int(end.Timestamp)))
                elif int(end.Timestamp)-int(srt.Timestamp)==2:
                    tmp.Timestamp = tmp.Timestamp+1
                    tmp = tmp.to_frame().T
                # paste it together
                eye_df  = pd.concat([pd.concat([eye_df.iloc[:end.index[0]], tmp], ignore_index=True), 
                                    eye_df.iloc[end.index[0]:]], ignore_index=True)
                del tmp
                del srt
                del end
            # make a distinction between first occurance and the rest
            tmp_bli = np.empty(len(eye_df))
            tmp_sac = np.empty(len(eye_df))
            tmp_fix = np.empty(len(eye_df))
            for i in range(1, len(eye_df)-1):
                if eye_df['Blink'][i]!=-1 and eye_df['Blink'][i-1]==-1: # start
                    tmp_bli[i] = eye['events']['eog/bli']
                elif eye_df['Blink'][i]!=-1 and eye_df['Blink'][i-1]!=-1: # continuation
                    tmp_bli[i] = eye['events']['eog/bli_c']
                if eye_df['SaccadeSeq'][i]!=-1 and eye_df['SaccadeSeq'][i-1]==-1: # start
                    tmp_sac[i] = eye['events']['eog/sac']
                elif eye_df['SaccadeSeq'][i]!=-1 and eye_df['SaccadeSeq'][i-1]!=-1: # continuation
                    tmp_sac[i] = eye['events']['eog/sac_c']
                if eye_df['FixationSeq'][i]!=-1 and eye_df['FixationSeq'][i-1]==-1: # start
                    tmp_fix[i] = eye['events']['eog/fix']
                elif eye_df['FixationSeq'][i]!=-1 and eye_df['FixationSeq'][i-1]!=-1: # continuation
                    tmp_fix[i] = eye['events']['eog/fix_c']
            eye_df['Blink_trg']    = tmp_bli.astype(int) # set trigger when there is a blink and to 0 when there is no blink
            eye_df['Saccade_trg']  = tmp_sac.astype(int) # set trigger when there is a saccade and to 0 when there is no saccade
            eye_df['Fixation_trg'] = tmp_fix.astype(int) # set trigger when there is a fixation and to 0 when there is no fixation
            eye_df.to_csv(var_loc['eye'])
        if not os.path.isfile(var_loc['raw']) and not os.path.isfile(var_loc['eog']):
            # LOAD THE DATA
            print('\nLOAD DATA\n')
            # if the data is collected in a continuous way (and split to bits by the MEG software), MNE will recognize the bits belonging to the same participant. If not, you need a loop to get all the files "manually"
            data = mne.io.read_raw_fif(os.path.join(paths["inp_MEG"],subj['nr'],subj['nr']+'.fif'),preload=True)
            # mark the unused channels as bad
            data.info['bads'] = ['ECG003', 'MISC001', 'MISC002', 'MISC003', 'MISC004', 'MISC005']
            # remove the unused channels
            data.drop_channels(ch_names = data.info['bads'])
            # save the data
            data.save(var_loc['raw'], overwrite=True)

        if not os.path.isfile(var_loc['bad']) and not os.path.isfile(var_loc['eog']):
            # if not in memory, load in the data
            if not 'data' in locals():
                data = mne.io.read_raw_fif(var_loc['raw'], preload=True) 
            #%% BAD SENSOR IDENTIFICATION
            print('\nIDENTIFY BAD SENSORS\n')
            # automatically check for faulty sensors
            if analysis['preproc']['cHPI']:
                print('\nHEAD-MOVEMENT CORRECTION (AND cHPI FILTER)\n')
                #%% cHPI
                # Compute time-varying cHPI amplitudes
                chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(
                                                                    raw = data
                                                                    )
                # get information of the data
                info = data.info
                # Compute locations of each cHPI coils over time
                chpi_locs = mne.chpi.compute_chpi_locs(
                                                        info            = info,
                                                        chpi_amplitudes = chpi_amplitudes
                                                        )
                # Compute time-varying head positions
                head_pos = mne.chpi.compute_head_pos(
                                                    info            = info,
                                                    chpi_locs       = chpi_locs
                                                    )
                head_pos_file = open(var_loc['hps'],'wb')
                pickle.dump(head_pos, head_pos_file)
                head_pos_file.close()
                auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
                                                            data, 
                                                            cross_talk      = os.path.join(paths['analysis'],meg['crosstalk_file']), 
                                                            calibration     = os.path.join(paths['analysis'],meg['calibration_file']),
                                                            return_scores   = True, 
                                                            head_pos        = head_pos
                                                            )
            if not analysis['preproc']['cHPI']:
                auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
                                                            data, 
                                                            cross_talk      = os.path.join(paths['analysis'],meg['crosstalk_file']), 
                                                            calibration     = os.path.join(paths['analysis'],meg['calibration_file']),
                                                            return_scores   = True, 
                                                            )
            # display the noisy/flat channels
            print('noisy =', auto_noisy_chs)
            print('flat = ', auto_flat_chs)
            # mark those channels as bad in the data
            data.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
            print('bads =', data.info['bads'])
            # load in the subject data
            with open(var_loc["sub"], "r", encoding="utf-8") as fp:
                subj = json.load(fp)
            subj['bads'] = [auto_noisy_chs + auto_flat_chs]
            # save the subject data
            with open(var_loc["sub"], "w", encoding="utf-8") as fp:
                json.dump(subj, fp, indent=2, default=to_json_safe)
            # fix magnetometer coil types (https://mne.tools/1.1/generated/mne.channels.fix_mag_coil_types.html)
            data.fix_mag_coil_types()
            if analysis['preproc']['intrp']:
                # INTERPOLATE BAD CHANNELS
                data = data.interpolate_bads()
            data.save(var_loc['bad'], overwrite=True)
            del [auto_noisy_chs, auto_flat_chs, auto_scores] 
        if not os.path.isfile(var_loc['sss']) and analysis['preproc']['Max'] and not os.path.isfile(var_loc['eog']):
            # if not in memory, load in the data
            if not 'data' in locals():
                data = mne.io.read_raw_fif(var_loc['bad'], preload=True) 
            # apply the algorithm performing the Maxfiltering, SSS, calibration and cross-talk reduction
            if analysis['preproc']['cHPI']:
                if not 'head_pos' in locals():   
                    with open(var_loc['hps'],'rb') as fp:
                        head_pos = pickle.load(fp)
                data_sss = mne.preprocessing.maxwell_filter(
                    data,
                    cross_talk      = os.path.join(paths['analysis'],meg['crosstalk_file']),
                    calibration     = os.path.join(paths['analysis'],meg['calibration_file']),
                    head_pos        = head_pos
                    )
            if not analysis['preproc']['cHPI']:
                data_sss = mne.preprocessing.maxwell_filter(
                    data,
                    cross_talk      = os.path.join(paths['analysis'],meg['crosstalk_file']),
                    calibration     = os.path.join(paths['analysis'],meg['calibration_file']),
                    )
            # replace the old data with the new data
            data=data_sss
            # save the data
            data.save(var_loc['sss'], overwrite=True)
            del data_sss
        if not os.path.isfile(var_loc['ano']) and not os.path.isfile(var_loc['eog']) or not os.path.isfile(var_loc['eve']):
            # if not in memory, load in the data
            if not 'data' in locals():
                if analysis['preproc']['Max']:
                    data = mne.io.read_raw_fif(var_loc['sss'], preload=True) 
                if not analysis['preproc']['Max']:
                    data = mne.io.read_raw_fif(var_loc['bad'], preload=True)
            #%% EVENT ADJUSTMENT
            print('\nADJUST EVENTS\n')
            # get the events
            events = mne.find_events(data, stim_channel=meg['chan_trg'], min_duration=0.001001, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)))
            idx = events[:,2]
            # transform the task events into a dataframe
            tmp = pd.DataFrame(task['org_events'],index=[0])
            # loop over the events to add the prime condition info to the foveal image onsets
            for e in range(len(events)):
                # if the current event is a foveal one..
                if np.isin(events[e,2],tmp.iloc[0][fnmatch.filter(tmp,'*/fov*')]):
                    #.. find the previous parafoveal stimulus and get the priming condition
                    # this should be the event 2 (or 4) events back
                    if np.isin(events[e-2,2],tmp.iloc[0][fnmatch.filter(tmp,'*s2/p0*')]):
                        events[e,2] = int(str(events[e,2])+'0')
                    elif np.isin(events[e-2,2],tmp.iloc[0][fnmatch.filter(tmp,'*s2/p1*')]):
                        events[e,2] = int(str(events[e,2])+'1')
                    elif np.isin(events[e-2,2],tmp.iloc[0][fnmatch.filter(tmp,'*s2/p2*')]):
                        events[e,2] = int(str(events[e,2])+'2')
            # save the events
            mne.write_events(var_loc['eve'], events, overwrite=True)
            print('\nANNOTATIONS\n')
            #%% BREAK ANNOTATION
            # https://mne.tools/dev/auto_tutorials/preprocessing/20_rejecting_bad_data.html
            annotations_break = mne.preprocessing.annotate_break(
                                                                raw                     = data,
                                                                events                  = events,
                                                                min_break_duration      = 5, # consider segments of at least 5 s duration
                                                                t_start_after_previous  = 2, # start annotation 2 s after end of previous one
                                                                t_stop_before_next      = 2  # stop annotation 2 s before beginning of next one
            )
            data.set_annotations(data.annotations + annotations_break)  # add to existing
            #%% MUSCLE ARTIFACT ANNOTATION
            # automatically detect muscle artifacts
            threshold_muscle = 10 # in z-scores # FLUX said 10, MNE tutorial 5 (https://mne.tools/stable/auto_examples/preprocessing/muscle_detection.html)
            annotations_muscle, scores_muscle = annotate_muscle_zscore(
                                                                    data, 
                                                                    ch_type             = "mag", 
                                                                    threshold           = threshold_muscle, 
                                                                    min_length_good     = 0.2,
                                                                    filter_freq         = [110, 140])
            # mark the break and muscle annotations in the data
            data.set_annotations(data.annotations + annotations_muscle)
            # plot the data
            fig = data.plot(
                            start=1000,
                            duration=20,
                            title='Annotations',
                            lowpass=40,
                            show= False)
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_annotations.png'))
            # close all figures
            plt.close('all')   
            # save the events
            mne.write_events(var_loc['eve'], events, overwrite=True)
            # save the data
            data.save(var_loc['ano'], overwrite=True)
        if not os.path.isfile(var_loc['eog']):
            # if not in memory, load in the eye-movement related triggers
            if not 'eye_df' in locals():
                eye_df = pd.read_csv(var_loc['eye'])
            # if not in memory, load in the data
            if not 'data' in locals():
                data = mne.io.read_raw_fif(var_loc['ano'], preload=True)  
            # if not in memory, load in the events
            if not 'events' in locals():   
                events = mne.read_events(var_loc['eve'])
            #%% MAKE EOG CHANNEL (FROM THE EYELINK DATA)
            print('\nADD EOG CHANNEL\n')
            # check if the "start of task marker is present in the Eyelink data
            if np.isin(task['org_events']['pf/task'],np.unique(eye_df.StimulusName)):
                # if it is construct EOG channel from the Eyelink data
                # get the Eyelink start of the task sample
                inds = eye_df[eye_df.StimulusName==task['org_events']['pf/task']]
                t_eye = eye_df.Timestamp[inds.first_valid_index()]
                # get the MEG start of the task sample
                inds = events[events[:,2]==task['org_events']['pf/task'],0]
                t_meg = inds[0]
                # Adjust for the difference between the two and padding the data
                # since the sampling frequency of the two is same(-ish):
                t_diff = t_meg - t_eye
                eye_df.Timestamp = eye_df.Timestamp + t_diff
                if data.first_samp < int(eye_df['Timestamp'].iat[0]): # MEG data recoding started before Eyelink
                    # create an empty matrix
                    tmp = np.zeros((int(eye_df['Timestamp'].iat[0]) - data.first_samp, eye_df.shape[1]))
                    # convert it to a dataframe
                    tmp = pd.DataFrame(tmp, columns=eye_df.columns)
                    # add the timestamps
                    tmp.Timestamp=range(data.first_samp,int(eye_df['Timestamp'].iat[0]))
                    # append it to the original Eyelink dataframe
                    eye_df = pd.concat([tmp,eye_df],ignore_index=True)
                    del tmp
                if data.first_samp > int(eye_df['Timestamp'].iat[0]): # MEG data recoding started after Eyelink
                    # cut the Eyelink data before the MEG recording started
                    eye_df = eye_df[eye_df.Timestamp>=data.first_samp]
                if data.last_samp > int(eye_df['Timestamp'].iat[len(eye_df.Timestamp)-1]): # MEG data recoding ended afer Eyelink
                    # create an empty matrix
                    tmp = np.zeros((data.last_samp+1 - int(eye_df['Timestamp'].iat[len(eye_df.Timestamp)-1]+1), eye_df.shape[1]))
                    # convert it to a dataframe
                    tmp = pd.DataFrame(tmp, columns=eye_df.columns)
                    # add the timestamps
                    tmp.Timestamp=range(int(eye_df['Timestamp'].iat[len(eye_df.Timestamp)-1])+1, data.last_samp+1)
                    # append it to the original Eyelink dataframe
                    eye_df = pd.concat([eye_df,tmp],ignore_index=True)
                    del tmp   
                if data.last_samp < int(eye_df['Timestamp'].iat[len(eye_df.Timestamp)-1]): # MEG data recoding ended before Eyelink
                    # cut the Eyelink data after the MEG recording stopped
                    eye_df = eye_df[eye_df.Timestamp<=data.last_samp]
                # Since the sampling frequency is not identical, adjust for the size, since MNE likes equal sizes
                if len(eye_df) < len(data):
                    # create an empty matrix
                    tmp = np.zeros((len(data) - len(eye_df), eye_df.shape[1]))
                    # convert it to a dataframe
                    tmp = pd.DataFrame(tmp, columns=eye_df.columns)
                    # add the timestamps
                    tmp.Timestamp=range(int(eye_df['Timestamp'].iat[len(eye_df.Timestamp)-1])+1, int(eye_df['Timestamp'].iat[len(eye_df.Timestamp)-1])+1+len(tmp))
                    # append it to the original Eyelink dataframe
                    eye_df = pd.concat([eye_df,tmp],ignore_index=True)
                # Adjust the data range
                eye_df.GazeLefty = eye_df.GazeLefty/eye['eog_adj']
                eye_df.GazeLeftx = eye_df.GazeLeftx/eye['eog_adj']
                # Add EOG channels
                # https://github.com/mne-tools/mne-python/issues/4208
                info = mne.create_info(['VEOG'], eye['samp_freq'], ['eog'])
                VEOG = mne.io.RawArray(eye_df.GazeLefty.to_numpy()[np.newaxis], info, eye_df.Timestamp[eye_df.Timestamp.index[0]])
                info = mne.create_info(['HEOG'], eye['samp_freq'], ['eog'])
                HEOG = mne.io.RawArray(eye_df.GazeLeftx.to_numpy()[np.newaxis], info, eye_df.Timestamp[eye_df.Timestamp.index[0]])
                data.add_channels([VEOG,HEOG],force_update_info=True)
                # make the blink events
                events_bli = np.transpose(np.array([np.array(eye_df.Timestamp).astype(int),np.empty(len(eye_df.Blink_trg)).astype(int),np.array(eye_df.Blink_trg).astype(int)]))
                events_bli = events_bli[events_bli[:,2]==eye['events']['eog/bli']]
                events_sac = np.transpose(np.array([np.array(eye_df.Timestamp).astype(int),np.empty(len(eye_df.Saccade_trg)).astype(int),np.array(eye_df.Saccade_trg).astype(int)]))
                events_sac = events_sac[events_sac[:,2]==eye['events']['eog/sac']]
                events_fix = np.transpose(np.array([np.array(eye_df.Timestamp).astype(int),np.empty(len(eye_df.Fixation_trg)).astype(int),np.array(eye_df.Fixation_trg).astype(int)]))
                events_fix = events_fix[(events_fix[:,2]==eye['events']['eog/fix']) | (events_fix[:,2]==eye['events']['eog/fix_c'])]
                events_eog = np.concatenate((events_bli,events_sac,events_fix))
                # mark the original EOG channels as bad
                data.info['bads'].extend(['EOG001', 'EOG002'])
                # plot the eye events
                fig = data.plot(
                                start       = 1000,
                                duration    = 20,
                                title       = 'Blinks',
                                lowpass     = 40,
                                events      = events_bli,
                                show        = False)
                fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_eve_blinks.png'))
                plt.close('all')
                #%% EYES CLOSED ANNOTATION
                # find the places where fixation was lost for more than a second ( = one time the sampling frequency)
                events_clo = events_fix[np.concatenate([np.diff(events_fix[:,0],1)>meg['samp_freq'],[False]]),:]
                # get those time points in seconds (samples / sampling frequency)
                times_clo = events_clo[:,0]/meg['samp_freq'] #  time in seconds
                # get the durations of the loses of fixation in seconds (samples / sampling frequency)
                diffs = np.diff(events_fix[:,0],1)
                dur_clo = diffs[diffs>meg['samp_freq']]/meg['samp_freq']
                # annotate the periods of time where fixation was gone for longer than one second ( = one time the sampling frequency)
                annotations_clo = mne.Annotations(
                                                onset                   = times_clo,
                                                duration                = dur_clo,
                                                description             = 'bad_fixation',
                                                orig_time               = data.info['meas_date']
                                                )
                # add to existing annotations
                data.set_annotations(data.annotations + annotations_clo)  
                # save the blink events
                np.save(var_loc['evb'],events_bli)
                # save the saccade events
                np.save(var_loc['evs'],events_sac)
                #%% BLINK ANNOTATION
                # mark the blinks that are in the first 100 ms of stimulus presentation
                if not 'events_bli' in locals(): 
                    try:
                        events_bli = np.load(var_loc['evb'])
                        # find the stimulus onset events (THIS WON'T WORK ANYMORE BECAUSE OF THE LATERALIZATION ADDED TO THE CODE AT THE EPOCHING STAGE DURING THE REVISION. TO MAKE THIS WORK THAT CODE NEEDS TO BE TRANSFERED TO THE EVENT ADJUSTMENT ABOVE)
                        events_sti = events[np.isin(events[:,2], [value for (key,value) in task['new_events'].items() if any( x in key for x in ['/fov','/pfov'])])]
                        # check if within the first 100 ms after stimulus onset there is a blink
                        for s in range(len(events_sti)):
                            # get the samples between stimulus onset and .1 seconds (100 ms) after
                            dur = .1
                            bad_bli = events_bli[np.isin(events_bli[:,0], range(events_sti[s,0],events_sti[s,0]+int(meg['samp_freq']*dur)))]
                            if bad_bli.any():
                                print(bad_bli)
                                # if there is a blink in the given timewindow, annotate it
                                annotations_bli = mne.Annotations(
                                                                onset                   = events_sti[s,0],
                                                                duration                = dur,
                                                                description             = 'bad_blink',
                                                                orig_time               = data.info['meas_date']
                                                                )
                        # add to existing annotations
                        data.set_annotations(data.annotations + annotations_bli) 
                    except:
                        print('Could not make blink annotations')
            else:
                # if not, construct VEOG channel from frontal sensors
                tmp = data.copy().filter(l_freq=2, h_freq=15) # filter based on: https://www.fieldtriptoolbox.org/tutorial/automatic_artifact_rejection/#detection-of-eog-artifacts
                tmp.pick(['MEG0312','MEG0313','MEG0342','MEG0343','MEG0122','MEG0123','MEG0112','MEG0113',
                          'MEG1212','MEG1213','MEG1222','MEG1223','MEG1412','MEG1413','MEG1422','MEG1423'])
                # get the average of the absolute values of the frontal channels
                tmp._data = np.mean(np.absolute(tmp._data),0)
                info = mne.create_info(['VEOG'], eye['samp_freq'], ['eog'])
                VEOG = mne.io.RawArray(tmp._data[np.newaxis]*1e8, info, data.first_samp)
                data.add_channels([VEOG], force_update_info=True)
                # mark the original EOG channels as bad
                data.info['bads'].extend(['EOG001', 'EOG002'])
            # save the data
            data.save(var_loc['eog'], overwrite=True)
        if not os.path.isfile(var_loc['ica']):
            print('\nCOMPUTE ICA COMPONENTS\n')
            # if not in memory, load in the data
            if not 'data' in globals():
                data = mne.io.read_raw_fif(var_loc['eog'], preload=True)  
            # BANDPASS FILTER THE DATA
            # filter the data to make it better for ICA
            data_resmpl = data.copy()
            data_resmpl.resample(250) # FLUX said 200
            data_resmpl.filter(analysis['bp_fil_ica']['l_freq'], analysis['bp_fil_ica']['h_freq']) # FLUX said 1-40 Hz
            #%% EOG & ECG SUPRESSION WITH ICA
            # set the ICA parameters
            ica = ICA(
                    method          = 'fastica', # FLUX
                    random_state    = 97, # A seed for the random number generator so that results are reproducible.
                    n_components    = 30, # FLUX said 30
                    verbose         = True)
            # fit the ICA
            ica.fit(
                    data_resmpl,
                    reject_by_annotation    = True, # Omit bad segments from the data before fitting
                    verbose                 = True)
            fig = ica.plot_sources(
                                    inst            = data_resmpl, 
                                    start           = 1000,
                                    #stop            = 1500,
                                    title           = 'ICA',
                                    show            = False,
                                    picks           = list(range(0,15))
                                    )
            fig = ica.plot_sources(
                                    inst            = data_resmpl, 
                                    start           = 1000,
                                    #stop            = 1500,
                                    title           = 'ICA',
                                    show            = False,
                                    picks           = list(range(15,30))
                                    )
            fig = ica.plot_components(show= False)
            save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_ICAcomps.pdf'))
            plt.close('all')
            # save the data
            ica.save(var_loc['ica'], overwrite=True)
        if not os.path.isfile(var_loc['pic']):
            print('\nREMOVE EOG (AND ECG) ICA COMPONENTS\n')
            # if not in memory, load in the data
            if not 'data' in locals():
                data = mne.io.read_raw_fif(var_loc['eog'], preload=True)
            if not 'ica' in locals():
                ica = mne.preprocessing.read_ica(var_loc['ica'])
            # if not in memory, load in the events
            if not 'events' in locals():   
                events = mne.read_events(var_loc['eve'])
            if not 'events_bli' in locals(): 
                try:
                    events_bli = np.load(var_loc['evb'])
                except:
                    print('Could not load the blink events')
            if not 'events_sac' in locals(): 
                try:
                    events_sac = np.load(var_loc['evs'])
                except:
                    print('Could not load the saccade events')
            # START THE AUTOMATIC/MANUAL DETECTION
            # EOG
            # check if there is a dictionary with the manual IC components and if we want to use them
            # !! be mindful that if the ica analysis is done again, the manual check has to be done again !!
            if os.path.isfile(os.path.join(paths["outp_MEG"],analysis['type'],'IC_manual','IC_EOG_'+analysis['type']+'_dict.pkl')) and analysis['ica_mnl']:
                # load them in
                IC_dir = os.path.join(paths["outp_MEG"],analysis['type'],'IC_manual')
                with open(os.path.join(IC_dir,'IC_EOG_'+analysis['type']+'_dict.pkl'), 'rb') as f:
                    IC_EOG_SVM_dict = pickle.load(f)
                with open(os.path.join(IC_dir,'IC_ECG_'+analysis['type']+'_dict.pkl'), 'rb') as f:
                    IC_ECG_SVM_dict = pickle.load(f)
                # store the indices
                ecg_inds = IC_EOG_SVM_dict[int(subj['nr'])]
                veog_inds = IC_ECG_SVM_dict[int(subj['nr'])]
                heog_inds = []
            else:
                #%% FILTER THE DATA
                # filter the data to make it better for EOG detection
                data_resmpl = data.copy()
                data_resmpl.filter(analysis['bp_fil_ica']['eog_l_freq'], analysis['bp_fil_ica']['eog_h_freq'])
                # remove the bad channels
                data_resmpl.drop_channels(
                                            ch_names = data_resmpl.info['bads']
                                          )
                # automatically detect blinks by correlating the EOG epochs with the ICA components
                # https://jasmainak.github.io/mne-workshop-brown/preprocessing/ica.html
                # VEOG
                veog_inds = []
                if 'events_bli' in locals():
                    veog_epochs = mne.Epochs(
                                              raw                    = data_resmpl,
                                              events                 = events_bli,
                                              tmin                   = -.2,
                                              tmax                   = .3,
                                              event_id               = eye['events']['eog/bli'],
                                              proj                   = True, #default option
                                              reject_by_annotation   = True,
                                              preload                = True
                                              )
                    veog_avg = veog_epochs.average()
                else:
                    if subj['nr']=='101':
                        veog_epochs = mne.preprocessing.create_eog_epochs(
                                                                        raw         = data_resmpl,
                                                                        ch_name     = 'VEOG',
                                                                        tmin        = -.2,
                                                                        tmax        = .3,
                                                                        thresh      = 40e-5) # 400 mircovolt
                    elif subj['nr']=='102':
                        veog_epochs = mne.preprocessing.create_eog_epochs(
                                                                        raw         = data_resmpl,
                                                                        ch_name     = 'VEOG',
                                                                        tmin        = -.2,
                                                                        tmax        = .3,
                                                                        thresh      = 60e-5) # 600 mircovolt
                        os.system("pause")
                    elif subj['nr']=='110':
                        veog_epochs = mne.preprocessing.create_eog_epochs(
                                                                        raw         = data_resmpl,
                                                                        ch_name     = 'VEOG',
                                                                        tmin        = -.2,
                                                                        tmax        = .3,
                                                                        thresh      = 50e-5) # 500 mircovolt
                    else:
                        print('No blink events, so no epochs made')
                    veog_avg = veog_epochs.average()
                fig = veog_avg.plot_image(show = False)
                fig = veog_avg.plot_joint(show = False)
                save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_blinks.pdf'))
                plt.close('all')
                # find the VEOG ICA components
                try:
                    veog_inds, scores = ica.find_bads_eog(
                                                        inst        = veog_epochs, # this can probably also just be the full 'raw' data
                                                        ch_name     = ['VEOG'],
                                                        measure     = 'correlation',
                                                        threshold   = .2) 
                except: # in case there is no HEOG channel
                    veog_inds, scores = ica.find_bads_eog(
                                                        inst        = veog_epochs, # this can probably also just be the full 'raw' data
                                                        ch_name     = ['VEOG'],
                                                        measure     = 'correlation',
                                                        threshold   = .2)
                if len(veog_inds)==0: # if no EOG component is found try with the whole data
                    veog_inds, scores = ica.find_bads_eog(
                                                        inst        = data_resmpl, # the full 'raw' data
                                                        ch_name     = ['VEOG'],
                                                        measure     = 'correlation',
                                                        threshold   = .2)
                # since there are basically always blinks, if no blink component survives the threshold, remove the component with the highest score
                if len(veog_inds)==0:
                    veog_inds = [np.argmax(abs(scores))]
                ica.exclude = veog_inds
                fig = ica.plot_scores(
                                      scores, 
                                      title='correlations',
                                      show = False)
                fig = ica.plot_sources(
                                        veog_avg, 
                                        title='timecourse',
                                        show = False)
                #if len(veog_inds)!=0:
                fig = ica.plot_properties(
                                        veog_epochs, 
                                        picks       = veog_inds, 
                                        psd_args    = {'fmax': 35.},
                                        image_args  = {'sigma': 1.},
                                        show = False)
                fig = ica.plot_overlay(
                                        veog_avg, 
                                        exclude     = veog_inds,  
                                        title       = 'overlay',
                                        show        = False)  
                save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_ICA_VEOG.pdf'))
                plt.close('all')
                del veog_epochs
                del veog_avg
                # HEOG
                heog_inds = []
                if 'events_sac' in locals():
                    if len(events_sac)>1000:
                        print('select only ten percent of the HEOG data')
                        events_s = events_sac[1::10] # select every 10th element as there are a lot of saccades
                    else:
                        events_s  = events_sac # select all of them for this participant
                    heog_epochs = mne.Epochs(
                                              raw                    = data_resmpl,
                                              events                 = events_s,
                                              tmin                   = -.2,
                                              tmax                   = .3,
                                              event_id               = eye['events']['eog/sac'],
                                              proj                   = True, #default option
                                              detrend                = 1, # linear detrend
                                              reject_by_annotation   = True,
                                              preload                = True
                                              )
                    try:
                        heog_avg = heog_epochs.average()
                        fig = heog_avg.plot_image(show = False)
                        fig = heog_avg.plot_joint(show = False)
                        save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_saccades.pdf'))
                        plt.close('all')
                        # find the HEOG ICA components
                        heog_inds, scores = ica.find_bads_eog(
                                                            inst        = heog_epochs, # this can probably also just be the full 'raw' data
                                                            ch_name     = ['HEOG'],
                                                            measure     = 'correlation',
                                                            threshold   = .2)
                    
                        if len(heog_inds)==0: # if no EOG component is found try with the whole data
                            heog_inds, scores = ica.find_bads_eog(
                                                                inst        = data_resmpl, # the full 'raw' data
                                                                ch_name     = ['HEOG'],
                                                                measure     = 'correlation',
                                                                threshold   = .2)
                        ica.exclude = heog_inds
                        fig = ica.plot_scores(
                                              scores, 
                                              title='correlations',
                                              show = False)
                        fig = ica.plot_sources(
                                                heog_avg, 
                                                title='timecourse',
                                                show = False)
                        if len(heog_inds)!=0:
                            fig = ica.plot_properties(
                                                    heog_epochs, 
                                                    picks       = heog_inds, 
                                                    psd_args    = {'fmax': 35.},
                                                    image_args  = {'sigma': 1.},
                                                    show = False)
                            fig = ica.plot_overlay(
                                                    heog_avg, 
                                                    exclude     = heog_inds,  
                                                    title       = 'overlay',
                                                    show        = False)
                        save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_ICA_HEOG.pdf'))
                        plt.close('all')
                        del heog_epochs
                        del heog_avg
                    except:
                        print('no HEOG components created')
                        heog_inds = []
        
                # ECG
                # automatically detect hearbeats by correlating the ECG epochs with the ICA components
                # https://jasmainak.github.io/mne-workshop-brown/preprocessing/ica.html
                # https://mne.tools/dev/auto_tutorials/preprocessing/40_artifact_correction_ica.html#sphx-glr-auto-tutorials-preprocessing-40-artifact-correction-ica-py
                # filter the data to make it usable of ECG detection
                data_resmpl = data.copy()
                data_resmpl.filter(l_freq=analysis['bp_fil_ica']['l_freq'], h_freq=100) # 100 because of the cHPI noise
                # remove the bad channels
                data_resmpl.drop_channels(
                                            ch_names = data_resmpl.info['bads']
                                          )
                ecg_inds = []
                ecg_epochs = mne.preprocessing.create_ecg_epochs(
                                                                  raw                    = data_resmpl, 
                                                                  reject_by_annotation   = True
                                                                  )
                fig = ecg_epochs.plot_image(combine="mean")
                avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))
                fig = avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))
                fig = avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
                save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_heartbeats.pdf'))
                plt.close('all')
                # find the ECG ICA components
                ecg_inds, scores = ica.find_bads_ecg(
                                                      inst        = ecg_epochs,
                                                      method      = 'correlation',  
                                                      measure     = 'correlation',
                                                      threshold   = .2) 
                if len(ecg_inds)==0: # if no ECG component is found try with the ecg epochs
                    ecg_inds, scores = ica.find_bads_ecg(
                                                          inst        = data_resmpl,
                                                          method      = 'correlation',
                                                          measure     = 'correlation',
                                                          threshold   = .2) 
                ica.exclude = ecg_inds
                # barplot of ICA component "ECG match" scores
                fig = ica.plot_scores(
                                      scores, 
                                      title='correlations',
                                      show = False
                                      )
                # plot ICs applied to raw data, with ECG matches highlighted
                fig = ica.plot_sources(
                                    data_resmpl, 
                                    show_scrollbars     = False,
                                    picks               = list(range(0,15)))
                fig = ica.plot_sources(
                                    data_resmpl, 
                                    show_scrollbars     = False,
                                    picks               = list(range(15,30)))
                # # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
                fig = ica.plot_sources(avg_ecg_epochs, 
                                        title='timecourse',
                                        show = False
                                        )
                if bool(ecg_inds):
                    # plot diagnostics
                    fig = ica.plot_properties(
                                            ecg_epochs, 
                                            picks       = ecg_inds, 
                                            psd_args    = {'fmax': 35.},
                                            image_args  = {'sigma': 1.},
                                            show        = False)
                    fig = ica.plot_overlay(
                                        avg_ecg_epochs, 
                                        exclude     = ecg_inds, 
                                        show        = False, 
                                        title       = 'overlay')
                save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_ICA_ECG.pdf'))
                plt.close('all')
                del ecg_epochs
                del avg_ecg_epochs
            
            print('\nVEOG ICAs: ' + str(veog_inds))
            print('\nHEOG ICAs: ' + str(heog_inds))
            print('\nECG ICAs: ' + str(ecg_inds))
            subj['ecg_inds']    = ecg_inds
            subj['veog_inds']   = veog_inds
            subj['heog_inds']   = heog_inds
            
            # filter the data to make it better for plotting
            data_resmpl = data.copy()
            data_resmpl.filter(analysis['bp_fil_ica']['l_freq'], analysis['bp_fil_ica']['h_freq'])
            # remove the bad channels
            data_resmpl.drop_channels(
                                        ch_names = data_resmpl.info['bads']
                                      )
            # mark the selected ICs to be excluded
            ica.exclude = veog_inds + heog_inds + ecg_inds
            # plot ICs applied to raw data, with bad ICs highlighted
            fig = ica.plot_sources(
                                data_resmpl, 
                                start               = 1000,
                                show_scrollbars     = False,
                                picks               = list(range(0,15)))
            fig = ica.plot_sources(
                                data_resmpl, 
                                start               = 1000,
                                show_scrollbars     = False,
                                picks               = list(range(15,30)))
            # plot the effect of all the IC components removed
            fig = ica.plot_overlay(
                                data_resmpl, 
                                exclude     = veog_inds + heog_inds + ecg_inds, 
                                show        = False, 
                                title       = 'Signals before (red) and after (black) IC removal',
                                start       = 1000*1000,
                                stop        = 1020*1000,
                                picks       =['mag']
                                )
            fig = ica.plot_overlay(
                                data_resmpl, 
                                exclude     = veog_inds + heog_inds + ecg_inds, 
                                show        = False, 
                                title       = 'Signals before (red) and after (black) IC removal',
                                start       = 1000*1000,
                                stop        = 1020*1000,
                                picks       =['grad']
                                )
            fig = data.plot(
                            start       = 1000,
                            duration    = 20,
                            title       = 'PreICA',
                            n_channels  = len(data.ch_names),
                            butterfly   = True,
                            show        = False,
                            show_scrollbars = False,
                            highpass    = 1,
                            lowpass     = 30)
            
            # ICA backprojection to the MEG data, excluding the EOG and ECG components, and save the IC indexes
            ica.apply(
                    data, 
                    exclude = veog_inds + heog_inds + ecg_inds)
            fig = data.plot(
                            start       = 1000,
                            duration    = 20,
                            title       = 'PostICA',
                            n_channels  = len(data.ch_names),
                            butterfly   = True,
                            show        = False,
                            show_scrollbars = False,
                            highpass    = 1,
                            lowpass     = 30)
            save_multi_image(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_postICA.pdf'))
            # save the data
            data.save(var_loc['pic'], overwrite=True)
            # save the subject data
            with open(var_loc["sub"], "w", encoding="utf-8") as fp:
                json.dump(subj, fp, indent=2, default=to_json_safe)
        # load in the subject data
        with open(var_loc["sub"], "r", encoding="utf-8") as fp:
            subj = json.load(fp)
        if not os.path.isfile(var_loc['epo']):
        # if cur_redo:
            print('\nMAKE EPOCHS\n')
            #%% Extracting condition-specific trials %%
            # if not in memory, load in the data
            if not 'data' in locals():
                data = mne.io.read_raw_fif(var_loc['pic'], preload=True)  
            if not 'events' in locals():
                events = mne.read_events(var_loc['eve'])  
            if subj['nr'] == '123':
                print('\nParticipant 123: Remove the last trial\n')
                events = events[events[:,0]<5998858,:] # remove the errorneous last trigger
            
            # # adjust the events to add laterality of the primes (THIS IDEALLY SHOULD HAVE BEEN DONE EARLIER IN THE CODE, BUT REVISION STAGE, AND THEREFORE SHOULD ONLY BE RAN ONCE HERE)
            # pattern = os.path.join(paths['inp_BEH'], f'Parafov_{subj["nr"]}*.csv')
            # matching_files = glob(pattern)
            # if matching_files:
            #     behav_parafov = pd.read_csv(matching_files[0])
            # else:
            #     raise FileNotFoundError(f"No file matching pattern '{pattern}' found in {paths['inp_BEH']}")
            
            # if subj['nr']=='111':
            #     events = events[events[:,0]<5781357,:] # remove the errorneous last trigger
            #     # Trials 721–1440 first
            #     part1 = behav_parafov[(behav_parafov["trial"] >= 721) & (behav_parafov["trial"] <= 1440)]
            #     # Trials 1–600 next
            #     part2 = behav_parafov[(behav_parafov["trial"] >= 1) & (behav_parafov["trial"] <= 600)]
            #     # Concatenate in the desired order
            #     behav_parafov = pd.concat([part1, part2], ignore_index=True)

            # behav_parafov[['targetPrimedAt', 'primeScreenSide']] = behav_parafov.apply(find_prime_info, axis=1)

            # fov_4dig = [int(str(num)[:4]) for num in list(dict(filter(lambda item:'/fov' in item[0], task['new_events'].items())).values())]
            # p_fov_4dig = [int(str(num)[:4]) for num in list(dict(filter(lambda item: '/p1' in item[0] or '/p2' in item[0], task['new_events'].items())).values())]
            
            # # event_id = task['new_events']
            # # fov_4dig_to_key = {}

            # # for key, val in event_id.items():
            # #     base_code = int(str(val)[:4])
            # #     if base_code in fov_4dig and '/fov' in key:
            # #         fov_4dig_to_key[base_code] = key

            # events_new = events
            
            # cnt = 0
            # for e in range(len(events)):
            #     # adjust it in a quick and dirty way 
            #     if np.isin(events[e,2],fov_4dig):
            #         # print(cnt)
            #         # print(fov_4dig_to_key[events[e,2]])
            #         if np.isin(events[e, 2], p_fov_4dig):
            #             # print('')
            #             events_new[e,2] = int(str(events[e,2]) + str(int(behav_parafov['primeScreenSide'][cnt])))
            #         cnt+=1
            # # # save the events
            # # mne.write_events(var_loc['eve'], events, overwrite=True)

            # filter the data
            # lowpass filter the data at 150 Hz
            data_lp = {}
            data_lp = data.copy().filter(l_freq=.1,h_freq=150)
            epochs = mne.Epochs(
                raw                     = data_lp,
                events                  = events, 
                event_id                = dict(filter(lambda item:'/fov' in item[0], task['new_events'].items())), #get only the epochs based on the target stimulus onset
                tmin                    = -1.5 , 
                tmax                    = 1.5, 
                baseline                = None,
                proj                    = True, #default option
                picks                   = meg['chan_all'],
                detrend                 = 1, # linear detrend
                reject                  = analysis['rej'],
                reject_by_annotation    = True,
                preload                 = True
                )
            fig = epochs.plot_drop_log(show = False)
            subj['epochs_dropped'] = epochs.drop_log_stats()
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'],subj['nr'] + '_excl_epochs.png'))
            # downsample to reduce the computatation time
            epochs_rs = epochs.copy().resample(analysis['resmpl'])
            epochs_rs.save(var_loc['epo'], overwrite=True)
            # save the subject data
            with open(var_loc["sub"], "w", encoding="utf-8") as fp:
                json.dump(subj, fp, indent=2, default=to_json_safe)
            plt.close('all')
        if not os.path.isfile(var_loc['elk']):
            print('\nMAKE EYELINK EPOCHS\n')
            #%% Extracting condition-specific trials %%
            # if not in memory, load in the data
            if not 'data' in locals():
                data = mne.io.read_raw_fif(var_loc['pic'], preload=True)  
            if not 'events' in locals():
                events = mne.read_events(var_loc['eve'])  
            if subj['nr'] == '123':
                print('\nParticipant 123: Remove the last trial\n')
                events = events[events[:,0]<5998858,:] # remove the errorneous last trigger
            epochs = mne.Epochs(
                raw                     = data,
                events                  = events, 
                event_id                = dict(filter(lambda item:'/fov' in item[0], task['new_events'].items())), #get only the epochs based on the target stimulus onset
                tmin                    = -1.5 , 
                tmax                    = 1.5, 
                baseline                = None,
                proj                    = True, #default option
                picks                   = meg['chan_eog'],
                detrend                 = None, # linear detrend
                reject                  = None,
                reject_by_annotation    = True,
                preload                 = True
                )
            # downsample to reduce the computatation time
            epochs_rs = epochs.copy().resample(analysis['resmpl'])
            epochs_rs.save(var_loc['elk'], overwrite=True)
        #%% WRAP-UP
        # clear the variables (there is probably a better way)
        if 'eye_df' in locals():
            del eye_df
        if 'data' in locals():
            del data
        if 'events' in locals():
            del events
        if 'ica' in locals():
            del ica
        if 'events_bli' in locals():
            del events_bli 
        if 'epochs' in locals():
            del epochs
        if 'epochs_rs' in locals():
            del epochs_rs
        # clear stored files we don't need anymore
        for f in glob(var_loc['raw'][:-4]+'*.fif'):
            os.remove(f)
        for f in glob(var_loc['fil'][:-4]+'*.fif'):
            os.remove(f)
        for f in glob(var_loc['bad'][:-4]+'*.fif'):
            os.remove(f)
        for f in glob(var_loc['ano'][:-4]+'*.fif'):
            os.remove(f)

# create a function for the analyses
def MEG_analysis(vars,subj):
    # this function will perform all the analyses of the MEG analysis
    # input arguments:
    # vars = all the variable of the experiment in a dictionary
    #
    # get the variables out of the dictionary
    locals().update(vars)
    print('Subject:',subj['nr'])
    #%% SAVED VARIABLES LOCATIONS
    var_loc = {
                'eve'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_events.fif'), # where the events are saved
                'pic'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_postica.fif'), # where the post ICA data is saved
                'epo'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_epo.fif'), # where the epoched data is saved
                'elk'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_elk.fif'), # where the epoched data is saved
                'erfm'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ERF_m_{cond}.fif'), # where the ERF(grad) data is saved
                'erfg'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ERF_g_{cond}.fif'), # where the ERF(mag) data is saved
                'erfg2'     : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ERF_g2_{cond}.fif'), # where the ERF(mag) data is saved
                'erfg3'     : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ERF_g3_{cond}.fif'), # where the ERF(mag) data is saved
                'vepm'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_VEP_m.fif'), # where the VEP(grad) data is saved
                'vepg'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_VEP_g.fif'), # where the VEP(mag) data is saved
                'vepg2'     : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_VEP_g2.fif'), # where the VEP(mag) data is saved
                'vepg3'     : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_VEP_g3.fif'), # where the VEP(mag) data is saved
                'tfr'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_tfr_{cond}.fif'), # where the TFR data is saved
                'svm'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_svm_{prime}_{cat}.nc'), # where the SVM data is saved
                'sub'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json') # where the subject data is saved
                }
    #%% Event-related fields %%
    if analysis['types']['erf'] and analysis['type']=='ERFTFR' and not os.path.isfile(os.path.join(paths["outp_MEG"],analysis['type'], subj['nr'] + '_ERFs_PF.pdf')):
        print('\nERF ANALYSIS\n')
        if not 'epochs' in locals():
            epochs = mne.read_epochs(var_loc['epo'], preload=True) 
        # load in the subject data
        with open(var_loc["sub"], "r", encoding="utf-8") as fp:
            subj = json.load(fp)
        # check if the data can just be loaded in
        conds = ['p0','p1','p2'] # conditions of interest
        file_paths = []
        file_paths = file_paths + [var_loc['erfm'].format(cond=cond) for cond in conds]
        file_paths = file_paths + [var_loc['erfg'].format(cond=cond) for cond in conds]
        file_paths.append(var_loc['vepm'])
        file_paths.append(var_loc['vepg'])
        cur_redo = False # True for recalculate, False for load in old data
        if all(os.path.isfile(path) for path in file_paths) and not cur_redo:
            ERF_m = {}
            ERF_g = {}
            for cond in conds:
                ERF_m[cond] = mne.read_evokeds(var_loc['erfm'].format(cond=cond))[0]
                ERF_g[cond] = mne.read_evokeds(var_loc['erfg'].format(cond=cond))[0]
            ERF_m_vep = mne.read_evokeds(var_loc['vepm'])[0]
            ERF_g_vep = mne.read_evokeds(var_loc['vepg'])[0]
        else:
            # COMPUTE THE ERFS BY HAND
            # https://imaging.mrc-cbu.cam.ac.uk/meg/VectorviewDescription
            # filter the data to remove high frequency data
            epochs.filter(l_freq=None, h_freq=30)
            # pick all the MEG channels
            chns = [epochs.info['ch_names'][c] for c in mne.pick_channels_regexp(epochs.info['ch_names'],'MEG*')]
            # out of those select the magnetometers (MEGXXX1)
            chns_m = list(itertools.compress(chns,[c[-1]=='1' for c in chns]))
            # and select only those channels
            epochs_m = epochs.copy()
            epochs_m.pick(chns_m)
            # perform baseline correction 
            epochs_m.apply_baseline(analysis['erf']['basewin'])
            # average it to get the ERFs
            ERF_m = dict(p0=epochs_m['fov']['p0'].average(),
                          p1=epochs_m['fov']['p1'].average(),
                          p2=epochs_m['fov']['p2'].average())
            # out of those select the planar lattitude gradiometers (MEGXXX2)
            chns_g2 = list(itertools.compress(chns,[c[-1]=='2' for c in chns]))
            # and select only those channels
            epochs_g2 = epochs.copy()
            epochs_g2.pick(chns_g2)
            # perform baseline correction 
            epochs_g2.apply_baseline(analysis['erf']['basewin'])
            # average it to get the ERFs
            ERF_g2 = dict(p0=epochs_g2['fov']['p0'].average(),
                          p1=epochs_g2['fov']['p1'].average(),
                          p2=epochs_g2['fov']['p2'].average())
            # out of those select the planar longitude gradiometers (MEGXXX3)
            chns_g3 = list(itertools.compress(chns,[c[-1]=='3' for c in chns]))
            # and select only those channels
            epochs_g3 = epochs.copy()
            epochs_g3.pick(chns_g3)
            # perform baseline correction 
            epochs_g3.apply_baseline(analysis['erf']['basewin'])
            # average it to get the ERFs
            ERF_g3 = dict(p0=epochs_g3['fov']['p0'].average(),
                          p1=epochs_g3['fov']['p1'].average(),
                          p2=epochs_g3['fov']['p2'].average())
            # combine both types of gradiometers into one 
            ERF_g = copy.deepcopy(ERF_m)
            ERF_g['p0']._data = np.sqrt((ERF_g2['p0'].get_data()**2 + ERF_g3['p0'].get_data()**2)/2)
            ERF_g['p1']._data = np.sqrt((ERF_g2['p1'].get_data()**2 + ERF_g3['p1'].get_data()**2)/2)
            ERF_g['p2']._data = np.sqrt((ERF_g2['p2'].get_data()**2 + ERF_g3['p2'].get_data()**2)/2)
            # average it over conditions to get the VEPs
            ERF_m_vep       = epochs_m.average()
            ERF_g2_vep      = epochs_g2.average()
            ERF_g3_vep      = epochs_g3.average()
            ERF_g_vep       = ERF_m_vep.copy()
            ERF_g_vep._data = np.sqrt((ERF_g2_vep.get_data()**2 + ERF_g3_vep.get_data()**2)/2)
            
            # save the data
            for cond, epochs in ERF_m.items():
                epochs.save(var_loc['erfm'].format(cond=cond), overwrite=True)
            for cond, epochs in ERF_g.items():
                epochs.save(var_loc['erfg'].format(cond=cond), overwrite=True)
            for cond, epochs in ERF_g2.items():
                epochs.save(var_loc['erfg2'].format(cond=cond), overwrite=True)
            for cond, epochs in ERF_g3.items():
                epochs.save(var_loc['erfg3'].format(cond=cond), overwrite=True)
            ERF_m_vep.save(var_loc['vepm'].format(cond=cond), overwrite=True)
            ERF_g_vep.save(var_loc['vepg'].format(cond=cond), overwrite=True)
            ERF_g2_vep.save(var_loc['vepg2'].format(cond=cond), overwrite=True)
            ERF_g3_vep.save(var_loc['vepg3'].format(cond=cond), overwrite=True)

        if analysis['erf']['vep_calc']:
            ## CHECK THE VISUAL EVOKED POTENTIALs (VEP)
            # FIND THE CHANNELS WITH THE BIGGEST VEP
            # didn't really work on a subject level
            # https://www.sciencedirect.com/science/article/pii/B9780444640321000345#f0035
            # gradiometers
            # get the peak values per channel for the timepoints between 0 and 500 ms
            tms             = [t>.05 and t<=.2 for t in ERF_g_vep.times] 
            gradmax         = np.max(ERF_g_vep.data[:,tms],axis=1) 
            gradmean        = np.mean(ERF_g_vep.data[:,tms],axis=1)
            # and choose the 5 channels with the higest peak amplitude
            maxpeakchns_g   = [ERF_g_vep.ch_names[i] for i in np.flip(np.argsort(gradmax))][0:]
            # and choose the 5 channels with the largest mean amplitude
            maxmeanchns_g   = [ERF_g_vep.ch_names[i] for i in np.flip(np.argsort(gradmean))][0:5]
            # select the channels that overlap and save those
            subj['VEP_g']   = list(set(maxpeakchns_g) & set(maxmeanchns_g))
            # magnetometers
            # get the peak min() and max() values per channel for the timepoints between 0 and 500 ms
            tms             = [t>.05 and t<=.2 for t in ERF_m_vep.times] 
            magmax          = np.max(ERF_m_vep.data[:,tms],axis=1)
            magmin          = np.min(ERF_m_vep.data[:,tms],axis=1)
            magmean         = np.mean(ERF_m_vep.data[:,tms],axis=1)
            # and choose the 5 channels with the higest or lowest peak amplitude
            maxpeakchns_m   = [ERF_m_vep.ch_names[i] for i in np.flip(np.argsort(magmax))][0:5]
            minpeakchns_m   = [ERF_m_vep.ch_names[i] for i in np.argsort(magmin)][0:5]
            # and choose the 5 channels with the largest mean amplitude
            maxmeanchns_m   = [ERF_m_vep.ch_names[i] for i in np.flip(np.argsort(magmean))][0:5]
            minmeanchns_m   = [ERF_m_vep.ch_names[i] for i in np.argsort(magmean)][0:5]
            # select the channels that overlap and save those
            subj['VEP_m1']   = list(set(maxpeakchns_m) & set(maxmeanchns_m))
            subj['VEP_m2']   = list(set(minpeakchns_m) & set(minmeanchns_m))

            # plot the grand averages (vep)
            fig = ERF_g_vep.plot(picks=np.unique(meg['chan_RS_g']+meg['chan_RE_g']), gfp=True, titles=dict(mag=subj['nr'] + '_Gradiometers'))
            fig.set_size_inches((20, 10), forward=True) 
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], 'VEPs', subj['nr'] + '_ERF_grad.png'))
            fig = ERF_m_vep.plot(picks=np.unique(meg['chan_RS_ml']+meg['chan_RE_ml']), gfp=True, titles=dict(mag=subj['nr'] + '_Magnetometers (left)'))
            fig.set_size_inches((20, 10), forward=True)
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], 'VEPs', subj['nr'] + '_ERF_magl.png'))
            fig = ERF_m_vep.plot(picks=np.unique(meg['chan_RS_mr']+meg['chan_RE_mr']), gfp=True, titles=dict(mag=subj['nr'] + '_Magnetometers (right)'))
            fig.set_size_inches((20, 10), forward=True) 
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], 'VEPs', subj['nr'] + '_ERF_magr.png'))
            fig = ERF_g_vep.plot_topomap(times=[0.15, 0.4], average=[ 0.1, 0.2], show_names=True)
            fig.set_size_inches((30, 15), forward=True)
            fig.text(x=.5,y=.9,s=subj['nr'] + '_Gradiometers')
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], 'VEPs', subj['nr'] + '_topo_grad.png'))
            fig = ERF_m_vep.plot_topomap(times=[0.15, 0.4], average=[ 0.1, 0.2], show_names=True)
            fig.set_size_inches((30, 15), forward=True)
            fig.text(x=.5,y=.9,s=subj['nr'] + '_Magnetometers')
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], 'VEPs', subj['nr'] + '_topo_mag.png'))
            # saving the images
            save_multi_image(os.path.join(paths["outp_MEG"], analysis['type'], subj['nr'] + '_VEPs.pdf'))
            plt.close('all')

        fig = mne.viz.plot_compare_evokeds(
                                            ERF_g, 
                                            title    = 'ES: Evoked time course of the conditions (grad)',
                                            combine  = 'mean',
                                            vlines   = [0,.5], # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
                                            picks    = meg['chan_RS_g'],
                                            show     = False, #Show figure if True.
                                            )
        # the figure resizing doesn't work on BlueBear, so I added this loop
        try:
            fig[0].set_size_inches((20, 10), forward=True)
        except:
            print('Could not alter the size of the figure')
        fig = mne.viz.plot_compare_evokeds(
                                            ERF_m, 
                                            title    = 'ES: Evoked time course of the conditions (mag left)',
                                            combine  = 'mean',
                                            vlines   = [0,.5], # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
                                            picks    = meg['chan_RS_ml'],
                                            show     = False, #Show figure if True.
                                            )
        # the figure resizing doesn't work on BlueBear, so I added this loop
        try:
            fig[0].set_size_inches((20, 10), forward=True)
        except:
            print('Could not alter the size of the figure')
        fig = mne.viz.plot_compare_evokeds(
                                            ERF_m, 
                                            title    = 'ES: Evoked time course of the conditions (mag right)',
                                            combine  = 'mean',
                                            vlines   = [0,.5], # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
                                            picks    = meg['chan_RS_mr'],
                                            show     = False, #Show figure if True.
                                            )
        # the figure resizing doesn't work on BlueBear, so I added this loop
        try:
            fig[0].set_size_inches((20, 10), forward=True)
        except:
            print('Could not alter the size of the figure')
        fig = mne.viz.plot_evoked_topo(
                                        list(ERF_g.values()),
                                        title        = 'Topography of evoked responses: grad',
                                        vline        = [0,.5], # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
                                        show         = False, #Show figure if True.
                                        ) # To better interpret the planar gradiometers one can apply the root-mean-square operation in which the magnitude of the field for two orthogonal gradiometers are combined
        # the figure resizing doesn't work on BlueBear, so I added this loop
        try:
            fig.set_size_inches((20, 10), forward=True)
        except:
            print('Could not alter the size of the figure')  
        fig = mne.viz.plot_evoked_topo(
                                        list(ERF_m.values()),
                                        title        = 'Topography of evoked responses: mag',
                                        vline        = [0,.5], # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
                                        show         = False, #Show figure if True.
                                        ) # To better interpret the planar gradiometers one can apply the root-mean-square operation in which the magnitude of the field for two orthogonal gradiometers are combined
        # the figure resizing doesn't work on BlueBear, so I added this loop
        try:
            fig.set_size_inches((20, 10), forward=True)
        except:
            print('Could not alter the size of the figure') 
        fig = ERF_m['p0'].plot_topomap(
                                title       = 'p0 - Topographical map: magnetometers',
                                time_unit   = 'ms',
                                times       = [.25], # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
                                average     = [.501], # The time window (in seconds) around a given time point to be used for averaging.
                                show        = False #Show figure if True.
                                ) 
        # plotting the gradiometers doesn't work because of the manual baseline correction above.
        fig = ERF_g['p0'].plot_topomap(
                                title       = 'p0 - Topographical map: gradiometers',
                                time_unit   = 'ms',
                                times       = [.25], # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
                                average     = [.501], # The time window (in seconds) around a given time point to be used for averaging.
                                show        = False #Show figure if True.
                                ) 
        # the figure resizing doesn't work on BlueBear, so I added this loop
        try:
            fig.set_size_inches((20, 10), forward=True)
        except:
            print('Could not alter the size of the figure') 
        # saving the images
        save_multi_image(os.path.join(paths["outp_MEG"], analysis['type'], subj['nr'] + '_ERFs_PF.pdf'))
        plt.close('all')
        # save the subject data
        with open(var_loc["sub"], "w", encoding="utf-8") as fp:
            json.dump(subj, fp, indent=2, default=to_json_safe)
    #%% Time-frequency representations of power %%
    if analysis['types']['tfr'] and analysis['type']=='ERFTFR' and not os.path.isfile(os.path.join(paths["outp_MEG"],analysis['type'], subj['nr'] + '_TFR.pdf')):
        print('\nTFR ANALYSIS\n')
        # load in (again) as the data might be cut in the preceding ERF analysis
        epochs = mne.read_epochs(var_loc['epo'], preload=True) 
        # loop over the parafoveal conditions to compute the TFRs
        conds = ['p0','p1','p2'] # conditions of interest
        tfr_data = {}
        tfr_data_avg = {}
        for cond in conds:
            # compute Time-Frequency Representation (TFR) using DPSS tapers
            tfr_data_avg[cond] = mne.time_frequency.tfr_multitaper(
                                                                  inst               = epochs[cond],
                                                                  freqs              = analysis['tfr']['freqs'],
                                                                  n_cycles           = analysis['tfr']['n_cycles_R2'],
                                                                  time_bandwidth     = analysis['tfr']['time_win'],
                                                                  use_fft            = True,
                                                                  return_itc         = False,
                                                                  average            = True,
                                                                  decim              = 2,
                                                                  n_jobs             = -1,
                                                                  verbose            = True
                                                                  )
            print(analysis['tfr']['n_cycles_R2'])
            # save the data
            tfr_data_avg[cond].save(var_loc['tfr'].format(cond=cond), overwrite=True)
            fig = tfr_data_avg[cond].plot(
                                            tmin            = -1.0,
                                            tmax            = 0.5,
                                            baseline        = [-1,-0.7],
                                            mode            = 'percent',
                                            title           = 'Time-frequency resolution ' + cond + ' (Parieto-occitital channels)',
                                            combine         = 'mean' # Type of aggregation to perform across selected channels.
                                            )
            # add vertical lines
            # -1.600--.700 = fixation, -.700--.450 = prime, -.450--.350 = fixation, -.350--.100 = prime, -.100-0 = fixation, 0-.500 = target, .500-1.400 = fixation
            ax = fig[0].gca()
            ax.axvline(x=-.7,color='0'); ax.axvline(x=-.45,color='0'); ax.axvline(x=-.35,color='0'); ax.axvline(x=-.1,color='0'); ax.axvline(x=0,color='0'); ax.axvline(x=.5,color='0')
            ax.text(x=-.575, y=95, s='parafoveal (p2)', alpha=.5, horizontalalignment='center', fontsize=8)
            ax.text(x=-.225, y=95, s='parafoveal (p1)', alpha=.5, horizontalalignment='center', fontsize=8)
            ax.text(x= .250, y=95, s='foveal',          alpha=.5, horizontalalignment='center', fontsize=8)
            # the figure resizing doesn't work on BlueBear, so I added this loop
            try:
                fig[0].set_size_inches((15, 10), forward=True)
            except:
                print('Could not alter the size of the figure')
            fig = tfr_data_avg[cond].plot_topo(
                                    tmin            = 0,
                                    tmax            = 0.5,
                                    baseline        = [-1,-0.7],
                                    mode            = 'percent', 
                                    fig_facecolor   = 'w',
                                    font_color      = 'k',
                                    title           = 'Topography of TFR ' + cond,
                                    show            = False
                                    )
            # the figure resizing doesn't work on BlueBear, so I added this loop
            try:
                fig[0].set_size_inches((15, 10), forward=True)
            except:
                print('Could not alter the size of the figure')
            fig = tfr_data_avg[cond].plot_topomap(
                                            tmin            = 0,
                                            tmax            = 0.5, 
                                            baseline        = [-.5,-0.125], 
                                            mode            = 'percent',
                                            title           = 'Topographical map ' + cond,
                                            show            = False
                                            )
            # the figure resizing doesn't work on BlueBear, so I added this loop
            try:
                fig[0].set_size_inches((15, 15), forward=True)
            except:
                print('Could not alter the size of the figure')
        save_multi_image(os.path.join(paths["outp_MEG"], analysis['type'], subj['nr'] + '_TFR.pdf'))
        plt.close('all')
    #%% Classification using a support vector machine %%
    # load in the subject data
    with open(var_loc["sub"], "r", encoding="utf-8") as fp:
        subj = json.load(fp)
    if analysis['types']['mvpa'] and analysis['type']=='SVM' and (not os.path.isfile(var_loc['svm']) or not 'SVM_time' in subj.keys()):
        print('\nMVPA ANALYSIS\n')
        if not 'epochs' in locals():
            epochs = mne.read_epochs(var_loc['epo'], preload=True) 
        # crop the data to only the window of interest
        epochs.crop(
                    tmin            = -.85,
                    tmax            = .65,
                    include_tmax    = True)
        # if windowed SVM is done, this is the timewindow
        if analysis['mvpa']['twin']:
            timewin          = analysis['mvpa']['twin']
            sampwin          = epochs.info['sfreq']*timewin # sampling frequency times time window to get the sample window
            #     raise Exception('Please adjust either the timewindow or the sampling frequency, to make sure the number of samples in the siding time window is dividable by 2')
            if analysis['mvpa']['twin']:
                subj['SVM_time'] = epochs.times[int(sampwin/2):int(len(epochs.times)-sampwin/2)] # cut the data outside of the sliding time window
            else:
                subj['SVM_time'] = epochs.times
            
        #%% SET_UP FOR THE SPECIFIC (CATEGORY vs OTHERS) SVM CLASSIFIER 
        # transform X into a 2D array (samples x features) and standarize the data by demeaning and scaling to unit variance.
        clf = make_pipeline(
                            Vectorizer(), 
                            StandardScaler(), 
                            sklearn.svm.SVC(
                                kernel       = 'linear', 
                                class_weight = 'balanced',
                                random_state = 0,
                            ))
        if analysis['mvpa']['temgen']:
            # Temporal generalization is an extension of the decoding over time approach. It consists in evaluating whether the model estimated at a particular time instant accurately predicts any other time instant.
            time_ana = GeneralizingEstimator(
                                          clf, 
                                          n_jobs=-1, 
                                          scoring='roc_auc', 
                                          verbose=True)
        else:
            # use the features of X, targets y, and clf to discrimate the experimental conditions as a function of time (i.e., at which point they start to differ)
            time_ana = SlidingEstimator(
                                          clf, 
                                          n_jobs=-1, 
                                          scoring='roc_auc', 
                                          verbose=True)
        # loop over the object category conditions of interest and compare them to the other conditions
        for prime in task['primes']:
            for cat in task['cats']:
                cats = task['cats'].copy()
                cats.remove(cat)
                # select the markers for the current conditions of interest
                cond1 = [value for (key,value) in task['new_events'].items() if cat in key and prime in key] # marker of the first condition being classified
                cond2 = [value for (key,value) in task['new_events'].items() if any(x in key for x in cats)] # marker of the second condition being classified
                # check if old data can be loaded in              
                svm_file = var_loc['svm'].format(prime=prime, cat=cat)
                if os.path.exists(svm_file):
                    print(f"Loading existing SVM file: {svm_file}")
                    da = xr.load_dataarray(svm_file)
                else:
                    print(f"Computing SVM: prime={prime}, cat={cat}")
                    # X = feature matrix (trials x channels x time)
                    X = epochs[np.isin(epochs.events[:,2],cond1+cond2)].get_data(picks='meg')
                    print(X.shape) # trial, chan, time
                    if analysis['mvpa']['twin']:
                        print('Timewindowed SVM')
                        if analysis['mvpa']['concat']:
                            X_win = np.empty((X.shape[0],int(X.shape[1]*sampwin),X.shape[2])) # trial, chan x timewin, time
                            for t in range(int(sampwin/2), int(X.shape[2]-sampwin/2)):
                                X_win[:,:,t] = np.reshape(X[:,:,int(t-np.floor(sampwin/2)):int(t+np.ceil(sampwin/2))],(X.shape[0],int(X.shape[1]*sampwin)))
                            X=X_win[:,:,int(sampwin/2):int(X.shape[2]-sampwin/2)]
                        else:
                            X_win = np.empty(X.shape)
                            for t in range(int(sampwin/2), int(X.shape[2]-sampwin/2)):
                                X_win[:,:,t] = np.mean(X[:,:,int(t-np.floor(sampwin/2)):int(t+np.ceil(sampwin/2))],axis=2)
                            X=X_win[:,:,int(sampwin/2):int(X.shape[2]-sampwin/2)]
                    print(X.shape)
                    # y = target vector (codes for the conditions being compared)
                    y = np.empty(X[:,1,1].shape)
                    y[:] = np.NaN
                    tmp = epochs.events[np.isin(epochs.events[:,2],cond1+cond2),2]
                    y[np.isin(tmp,cond1)] = 1
                    y[np.isin(tmp,cond2)] = 2
                    # The classification will be performed timepoint by timepoint using a SVM by training on 80% of the trials on test on 20% in 5 runs. This results in a 5-fold cross-validation (cv=5). The output will be reported as Area Under the Curve (AUC).
                    score = cross_val_multiscore(
                        time_ana, 
                        X, 
                        y, 
                        cv=5, 
                        n_jobs=-1
                    )
                    # Mean scores across cross-validation splits
                    score = np.mean(score, axis=0)
                    # Saving
                    if analysis['mvpa']['temgen']:
                        da = xr.DataArray(
                            score,
                            dims=("time_train", "time_test"),
                            coords={
                                "time_train": subj["SVM_time"],
                                "time_test": subj["SVM_time"],
                                "prime": prime,
                                "cat": cat,
                                "subject": subj["nr"],
                            },
                            name="auc",
                        )
                    else:
                        da = xr.DataArray(
                            score,
                            dims=("time",),
                            coords={
                                "time": subj["SVM_time"],
                                "prime": prime,
                                "cat": cat,
                                "subject": subj["nr"],
                            },
                            name="auc",
                        )

                    da.to_netcdf(var_loc['svm'].format(prime=prime, cat=cat))

                    del cond1, cond2, X, y, tmp
                    if analysis['mvpa']['twin']:
                        del X_win

            # PLOTTING
            # load the data      
            das = [xr.load_dataarray(f).expand_dims("cat") for f in glob(var_loc['svm'].format(prime=prime, cat='*'))] 
            da_prime = xr.concat(das, dim="cat")
            # average over all categories per prime
            da_mean = da_prime.mean(dim="cat")
            
            # plot the results (on the diagonal)
            if analysis['mvpa']['temgen']:
                fig, (ax2,ax1) = plt.subplots(2,1,figsize=(7,10))
                fig.suptitle(task['pr_lab'][task['primes'].index(prime)], fontsize=16)
                ax1.plot(subj["SVM_time"], np.diag(da_mean.values))
            else:
                fig, ax1 = plt.subplots()
                ax1.plot(subj["SVM_time"], da_mean.values)
            
            plt.ylim([0.35, 0.75])
            plt.xlim(subj["SVM_time"][0], subj["SVM_time"][-1])
            
            ax1.axvline(x=-.7,color='0'); plt.axvline(x=-.45,color='0'); plt.axvline(x=-.35,color='0'); plt.axvline(x=-.1,color='0'); plt.axvline(x=0,color='0'); plt.axvline(x=.5,color='0')
            ax1.text(x=-.575, y=.6, s='parafoveal (p2)', alpha=.5, horizontalalignment='center', fontsize=8)
            ax1.text(x=-.225, y=.6, s='parafoveal (p1)', alpha=.5, horizontalalignment='center', fontsize=8)
            ax1.text(x= .250, y=.6, s='foveal',          alpha=.5, horizontalalignment='center', fontsize=8)
            ax1.axhline(.5, color='k', linestyle='--', label='chance')
            ax1.set_xlabel('Times')
            ax1.set_ylabel('AUC')  # Area Under the Curve
            ax1.legend()
            ax1.axvline(.0, color='k', linestyle='-')
            ax1.set_title('Sensor space decoding one vs others')
            
            if analysis['mvpa']['temgen']:
                # plot the full (generalization) matrix
                ax2.axvline(x=-.7,color='0'); ax2.axvline(x=-.45,color='0'); ax2.axvline(x=-.35,color='0'); ax2.axvline(x=-.1,color='0'); ax2.axvline(x=0,color='0'); ax2.axvline(x=.5,color='0')
                ax2.axhline(y=-.7,color='0'); ax2.axhline(y=-.45,color='0'); ax2.axhline(y=-.35,color='0'); ax2.axhline(y=-.1,color='0'); ax2.axhline(y=0,color='0'); ax2.axhline(y=.5,color='0')
                im = ax2.imshow(
                                da_mean.values,
                                interpolation="lanczos",
                                origin="lower",
                                cmap="RdBu_r",
                                extent=epochs.times[[0, -1, 0, -1]],
                                )
                ax2.set_xlabel("Testing Time (s)")
                ax2.set_ylabel("Training Time (s)")
                ax2.set_title("Temporal generalization")
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label("AUC")

            # save plot
            fig.savefig(
                os.path.join(paths["outp_MEG"], analysis['type'], subj['nr'] + '_SVM_' + prime + '.png'),
                dpi=300,
                bbox_inches="tight"
            )

        # combine plots
        save_multi_image(os.path.join(paths["outp_MEG"], analysis['type'], subj['nr'] + '_SVM_decoding_accuracy.pdf'))
        
        plt.close('all')
        
        # save the subject data
        with open(var_loc["sub"], "w", encoding="utf-8") as fp:
            json.dump(subj, fp, indent=2, default=to_json_safe)
    
#%% loop over the participants
print(sys.argv)
if "Users" in sys.argv[0]:
    for subj['nr'] in sorted(os.listdir(paths['inp_MEG']))[0:len(os.listdir(paths['inp_MEG']))]:
        # run the preprocessing function
        MEG_preproc(vars,subj)
        # run the analysis function
        MEG_analysis(vars,subj)
else:
    subj['nr'] = sys.argv[1]
    # run the preprocessing function
    MEG_preproc(vars,subj)
    # run the analysis function
    MEG_analysis(vars,subj)
    
    
#%% DATA CHECK
if analysis['types']['dc']:
    # setup
    svmdif_thres    = 5 # threshold for minimal percent of increase in AUC
    epdrp_thres     = 50 # threshold for maximum percent of the epochs dropped
    bad_ppn         = [] # manually selected as "bad"
    # placeholders
    mean_pre        = {}
    mean_post       = {}
    mean_diff       = {}
    epdrp           = {}
    # loop over the participants
    for subj['nr'] in sorted(os.listdir(paths['inp_MEG'])):
        if analysis['type'] == 'ERFTFR':
            var_loc = {
                        'sub' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json') # where the subject data is saved
                    }
        if analysis['type'] == 'SVM':
            var_loc = {
                        'svm' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_svm.obj'), # where the epoched data is saved
                        'sub' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json') # where the subject data is saved
                    }
            # get the SVM data
            with open(var_loc['svm'],'rb') as fp:
                svm_file = pickle.load(fp)
        with open(var_loc["sub"], "r", encoding="utf-8") as fp:
            subj = json.load(fp)
        # reset to good participant
        subj['bad']=False
        #%% Exclusion based on number of excluded epochs
        # get the number of dropped epochs
        epdrp[subj['nr']] = subj['epochs_dropped_' + analysis['type']]
        # if there are more bad trials than the threshold, mark this subject as bad
        if epdrp[subj['nr']]>=epdrp_thres:
            subj['bad'] = True
        #%% Exclusion based on manual exclusion
        if subj['nr'] in bad_ppn:
            subj['bad'] = True
        # save the subject data
        # subj_file = open(var_loc['sub'],'wb')
        # pickle.dump(subj, subj_file)
        # subj_file.close()
        with open(var_loc["sub"], "w", encoding="utf-8") as fp:
            json.dump(subj, fp, indent=2, default=to_json_safe)
    # print the bad subjects
    if analysis['type'] == 'SVM':
        print('\nParticipants with less than ' + str(svmdif_thres) + '% increase in SVM AUC after stimulus onset (no prime)')
        print(*[key for (key,value) in mean_diff.items() if value < mean_pre[key]*svmdif_thres/100], sep=", ")
    print('\nParticipants with more than ' + str(epdrp_thres) +'% of excluded trials')
    print(*[key for (key,value) in epdrp.items() if value >=epdrp_thres], sep=", ")
        
#%% GRAND AVERAGES
if analysis['types']['ga']:
    if analysis['types']['erf'] and analysis['type']=='ERFTFR':
        #%% ERFs
        print('\nGRAND AVERAGES: ERF ANALYSIS\n')
        var_loc = {
                    'gag'       : os.path.join(paths["outp_MEG"],analysis['type'],'ERF_g_ga_{cond}.fif'), # where the grand average data is saved
                    'gam'       : os.path.join(paths["outp_MEG"],analysis['type'],'ERF_m_ga_{cond}.fif'), # where the grand average data is saved
                    'gal'       : os.path.join(paths["outp_MEG"],analysis['type'],'ERF_g_all_{cond}_{subj}.fif'), # where the all subject data is saved
                    'mal'       : os.path.join(paths["outp_MEG"],analysis['type'],'ERF_m_all_{cond}_{subj}.fif'), # where the all subject data is saved
                    'vepgag'    : os.path.join(paths["outp_MEG"],analysis['type'],'ERF_g_VEP_ga.fif'), # where the VEP grand average data is saved
                    'vepgam'    : os.path.join(paths["outp_MEG"],analysis['type'],'ERF_m_VEP_ga.fif'), # where the VEP grand average data is saved
                  }
        cn_types = ['magl','magr','grad']
        cur_times = analysis['erf']['cbp_times']
        t_win = {
            'rs' : [0.15, 0.21], # repetition suppression
            're' : [0.30, 0.36] # repetition enhancement
        }
        # check if the grand average data can just be loaded in
        conds = task['primes'] # conditions of interest
        subj_names = []
        for s in sorted(os.listdir(paths['inp_MEG'])):
            sub_file = os.path.join(paths["outp_MEG"], analysis['type'], f"{s}_sub.json")
            if os.path.isfile(sub_file):
                with open(sub_file) as f:
                    info = json.load(f)
                if not info.get("bad", False):
                    subj_names.append(s)
        file_paths = []
        file_paths += [var_loc['gag'].format(cond=cond) for cond in conds]
        file_paths += [var_loc['gam'].format(cond=cond) for cond in conds]
        file_paths += [var_loc['gal'].format(cond=cond, subj=subj_name) for cond in conds for subj_name in subj_names]
        file_paths += [var_loc['mal'].format(cond=cond, subj=subj_name) for cond in conds for subj_name in subj_names]
        if analysis['erf']['vep_calc']:
            file_paths.append(var_loc['vepgag'])
            file_paths.append(var_loc['vepgam'])
        cur_redo = False # True for recalculate, False for load in old data
        if all(os.path.isfile(path) for path in file_paths) and not cur_redo:
            ERF_m_ga = {}
            ERF_g_ga = {}
            ERF_g_all = {}
            ERF_m_all = {}
            for cond in conds:
                ERF_g_ga[cond] = mne.read_evokeds(var_loc['gag'].format(cond=cond))[0]
                ERF_m_ga[cond] = mne.read_evokeds(var_loc['gam'].format(cond=cond))[0]
                ERF_g_all[cond] = []
                ERF_m_all[cond] = []
                for subj_name in subj_names: 
                    ERF_g_all[cond].append((
                        subj_name,
                        mne.read_evokeds(var_loc['gal'].format(cond=cond, subj=subj_name))[0]
                    ))
                    ERF_m_all[cond].append((
                        subj_name,
                        mne.read_evokeds(var_loc['mal'].format(cond=cond, subj=subj_name))[0]
                    ))
            if analysis['erf']['vep_calc']:
                ERF_g_vep_ga = mne.read_evokeds(var_loc['vepgag'])[0]
                ERF_m_vep_ga = mne.read_evokeds(var_loc['vepgam'])[0]

        else:
            ERF_m_all = {}
            ERF_g_all = {}
            ERF_g2_all = {}
            ERF_g3_all = {}
            ERF_m_ga = {}
            ERF_g_ga = {}
            ERF_g2_ga = {}
            ERF_g3_ga = {}
            ERF_m_vep_lst = []
            ERF_g_vep_lst = []
            ERF_g2_vep_lst = []
            ERF_g3_vep_lst = []
            for prime in task['primes']:
                ERF_m_all[prime]=[]
                ERF_g_all[prime]=[]
                ERF_g2_all[prime]=[]
                ERF_g3_all[prime]=[]
            # loop over participants
            for subj['nr'] in sorted(os.listdir(paths['inp_MEG'])):
                #%% SAVED VARIABLES LOCATIONS
                var_loc.update({
                                'epo'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_epo.fif'), # where the epoched data is saved
                                'erfm'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ERF_m_{cond}.fif'), # where the ERF(grad) data is saved
                                'erfg'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_ERF_g_{cond}.fif'), # where the ERF(mag) data is saved
                                'vepm'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_VEP_m.fif'), # where the VEP(grad) data is saved
                                'vepg'      : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_VEP_g.fif'), # where the VEP(mag) data is saved
                                'sub'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json') # where the subject data is saved
                                })
                # get the subject info         
                with open(var_loc["sub"], "r", encoding="utf-8") as fp:
                    subj = json.load(fp)

                # only add the subject to the grand average if it is not marked as bad
                if not subj['bad']:
                    # load in the data per participant  
                    ERF_m = {}
                    ERF_g = {}
                    subj['ERF'] = {}
                    for prime in task['primes']:
                        ERF_m[prime] = mne.read_evokeds(var_loc['erfm'].format(cond=prime))[0]
                        ERF_g[prime] = mne.read_evokeds(var_loc['erfg'].format(cond=prime))[0]
                        subj['ERF'][prime] = {}
                        # Save the data for the correlation with behavior
                        for key, (t_start, t_end) in t_win.items():
                            subj['ERF'][prime][key] = {}
                            cur_tms_mask = (ERF_g[prime].times > t_start - 0.001) & (ERF_g[prime].times < t_end + 0.001)
                            for chty in cn_types:
                                if chty=='grad':
                                    cur_data = ERF_g[prime].get_data()
                                    cur_chns_mask = np.isin(ERF_g[prime].ch_names, meg['chan_RS_g']+meg['chan_RE_g']) 
                                elif chty=='magl':
                                    ch_type  = 'mag'
                                    cur_data = ERF_m[prime].get_data()
                                    cur_chns_mask = np.isin(ERF_g[prime].ch_names, meg['chan_RS_ml']+meg['chan_RE_ml']) 
                                elif chty=='magr':
                                    ch_type  = 'mag'
                                    cur_data = ERF_m[prime].get_data()
                                    cur_chns_mask = np.isin(ERF_g[prime].ch_names, meg['chan_RS_mr']+meg['chan_RE_mr'])
                                selected_data = cur_data[cur_chns_mask][:, cur_tms_mask]
                                mean_over_channels = selected_data.mean(axis=0)
                                if chty=='grad':
                                    print(key, chty, 'max')
                                    val = mean_over_channels.max()
                                elif key=='rs' and chty=='magr':
                                    print(key, chty, 'max')
                                    val = mean_over_channels.max()
                                elif key=='rs' and chty=='magl':
                                    print(key, chty, 'min')
                                    val = mean_over_channels.min()
                                if key=='re' and chty=='magr':
                                    print(key, chty, 'min')
                                    val = mean_over_channels.min()
                                elif key=='re' and chty=='magl':
                                    print(key, chty, 'max')
                                    val = mean_over_channels.max()
                                subj['ERF'][prime][key][chty] = val
                    # Append the ERFs
                    for prime in task['primes']:
                          ERF_m_all[prime].append((subj['nr'], ERF_m[prime]))
                          ERF_g_all[prime].append((subj['nr'], ERF_g[prime]))
                    if analysis['erf']['vep_calc']:
                        # load in the data per participant
                        ERF_m_vep = mne.read_evokeds(var_loc['vepm'])[0]
                        ERF_g_vep = mne.read_evokeds(var_loc['vepg'])[0]
                        # Append the VEPs
                        ERF_m_vep_lst.append(ERF_m_vep)
                        ERF_g_vep_lst.append(ERF_g_vep)
                
                # save the subject data
                with open(var_loc["sub"], "w", encoding="utf-8") as fp:
                    json.dump(subj, fp, indent=2, default=to_json_safe)
   
            # make the grand averages
            for prime in task['primes']:
                ERF_m_ga[prime]             = mne.grand_average([ev for _, ev in ERF_m_all[prime]])
                ERF_m_ga[prime].comment     = prime + ' ' + ERF_m_ga[prime].comment
                ERF_g_ga[prime]             = mne.grand_average([ev for _, ev in ERF_g_all[prime]])
                ERF_g_ga[prime].comment     = prime + ' ' + ERF_g_ga[prime].comment
            if analysis['erf']['vep_calc']:
                ERF_m_vep_ga    = mne.grand_average(ERF_m_vep_lst)
                ERF_g_vep_ga    = mne.grand_average(ERF_g_vep_lst)
            
            # save the GA data
            for prime, evoked in ERF_g_ga.items():
                evoked.save(
                    var_loc['gag'].format(cond=prime),
                    overwrite=True
                )
            for prime, evoked in ERF_m_ga.items():
                evoked.save(
                    var_loc['gam'].format(cond=prime),
                    overwrite=True
                )

            for prime, pairs in ERF_g_all.items():
                for subj_name, ev in pairs:
                    ev.save(
                        var_loc['gal'].format(cond=prime, subj=subj_name),
                        overwrite=True
                    )

            for prime, pairs in ERF_m_all.items():
                for subj_name, ev in pairs:
                    ev.save(
                        var_loc['mal'].format(cond=prime, subj=subj_name),
                        overwrite=True
                    )

            if analysis['erf']['vep_calc']:
                ERF_g_vep_ga.save(
                    var_loc['vepgag'],
                    overwrite=True
                )
                ERF_m_vep_ga.save(
                    var_loc['vepgam'],
                    overwrite=True
                )

        # CLUSTER-BASED PERMUTATION
        # https://mne.tools/stable/auto_examples/stats/cluster_stats_evoked.html#sphx-glr-auto-examples-stats-cluster-stats-evoked-py
        # https://mne.tools/stable/auto_tutorials/stats-sensor-space/40_cluster_1samp_time_freq.html#sphx-glr-auto-tutorials-stats-sensor-space-40-cluster-1samp-time-freq-py
        # https://mne.discourse.group/t/how-to-set-up-a-cluster-permutation-test-between-two-independent-sample-groups/4668/2
        # https://mne.tools/stable/auto_examples/stats/sensor_permutation_test.html
        # https://mne.tools/stable/auto_examples/stats/sensor_permutation_test.html#sphx-glr-auto-examples-stats-sensor-permutation-test-py
        # loop over the various channel types (3)
        cbp_type = 'time' # 'time' or 'chanxtime'
        clus_results = {}
        cnt = 0
        plots = {}
        for chty in cn_types:
            # reshape the data for the CBP
            X = {}
            sel = {}
            cur_X = [None] * 3
            for prime in task['primes']:
                # placeholders
                X[prime]=[]
                sel[prime]=[]
                # select the channels
                if chty=='grad':
                    ch_type  = 'grad'
                    cur_data = ERF_g_all[prime]
                    cur_chns = meg['chan_RS_g']+meg['chan_RE_g']
                    cur_name = 'GA_ERFs_'+cbp_type+'_grad_perm.png'
                elif chty=='magl':
                    ch_type  = 'mag'
                    cur_data = ERF_m_all[prime]
                    cur_chns = meg['chan_RS_ml']+meg['chan_RE_ml']
                    cur_name = 'GA_ERFs_'+cbp_type+'_magl_perm.png'
                elif chty=='magr':
                    ch_type  = 'mag'
                    cur_data = ERF_m_all[prime]
                    cur_chns = meg['chan_RS_mr']+meg['chan_RE_mr']
                    cur_name = 'GA_ERFs_'+cbp_type+'_magr_perm.png'
                # extract a reference Evoked once
                _, ref_evoked = cur_data[0]
                # make the channel adacency used in the cluster-based permutation below
                if cbp_type == 'time':
                    adjacency = None
                elif cbp_type == 'chanxtime':
                    ch_adjacency, ch_names = mne.channels.find_ch_adjacency(
                                                info    = ref_evoked.info, 
                                                ch_type = ch_type)
                    ch_names = [s.replace(" ","") for s in ch_names] # remove the spaces to make it match
                    # since our data now is collapsed over the latitudinal and longitudinal gradiometers, and thus only contains half the channels, we need to only select channels ending in '2'
                    use_idx = [ch_names.index(ch_name) for ch_name in ref_evoked.ch_names]
                    ch_adjacency = ch_adjacency[use_idx][:, use_idx]
                    # prepare adjacency info for the time plane and combine
                    adjacency = mne.stats.combine_adjacency(
                        ch_adjacency, len(ref_evoked.times)
                        )
                # loop over subjects
                for i in range(len(cur_data)): 
                    _, evoked = cur_data[i]
                    tmp = evoked.get_data() # get the data for this subject (chan x time)
                    # # select only the relevant time window
                    if cbp_type == 'time':
                        cns = np.isin(evoked.ch_names,cur_chns) # select the channels
                        tmp = np.mean(tmp[cns,],0) # average over those channels
                    elif cbp_type == 'chanxtime':
                        pass
                    if i==0:
                        if cbp_type == 'time':
                            X[prime] = tmp
                        elif cbp_type == 'chanxtime':
                            X[prime] = [tmp]
                    else:
                        if cbp_type == 'time':
                            X[prime] = np.vstack([X[prime],np.reshape(tmp,(1,len(tmp)))]) # stack all participants underneath each other
                        elif cbp_type == 'chanxtime':
                            X[prime].append(tmp) # make 2d subarrays
                    # only test the data in the relevant time window
                    if cbp_type == 'time':
                        sel[prime] = (ref_evoked.times<cur_times[0]-.001) | (ref_evoked.times>cur_times[1]+.001) # selection of relevant time window
                    elif cbp_type == 'chanxtime':
                        sel[prime] = np.tile([(ref_evoked.times<cur_times[0]-.001) | (ref_evoked.times>cur_times[1]+.001)], (np.shape(X[prime][0])[0],1)) # selection of relevant time window
            # start running the actual cluster-based permutation
            X['z']=np.zeros(np.shape(X['p1'])) # to potentially test against 0
            if cbp_type == 'time':
                cur_X       = [X['p1']-X['p0'], X['p2']-X['p0']]
            elif cbp_type == 'chanxtime':
                cur_X[0]    = [a_i - b_i for a_i, b_i in zip(X['p0'], X['p1'])]
                cur_X[1]    = [a_i - b_i for a_i, b_i in zip(X['p0'], X['p2'])]
                cur_X[2]    = [a_i - b_i for a_i, b_i in zip(X['p2'], X['p1'])]
            cur_lab = ['p1-p0', 'p2-p0']
            cur_sel = sel['p0'] # we can just use the one from the first condition as they are all the same
            t_values=[]
            clus=[]
            clus_ps=[]
            for c in range(len(cur_X)):
                t_thresh = st.t.ppf(1 - analysis['erf']['cbp_clalpha'] / 2, df=X['z'].shape[0]-1) # cluster p = .005 (two-tailed)
                print(f"t_thresh = {round(t_thresh,3)}")
                T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                    X               = np.array(cur_X[c]),   # data to be clustered
                    n_permutations  = analysis['erf']['cbp_nperm'],                 # number of permutations to compute
                    tail            = 0,                    # two-tailed
                    adjacency       = adjacency,
                    threshold       = t_thresh,
                    seed            = 0,
                    exclude         = cur_sel,
                    out_type        = "mask",               # boolean arrays where true indicates part of a cluster
                )
                # save the data in list
                t_values.append(T_obs)
                clus.append(clusters)
                clus_ps.append(cluster_p_values)
                # Print significant clusters
                significant_clusters = np.where(cluster_p_values < 0.1)[0]
                nonsig_clusters = np.where(cluster_p_values >= 0.1)[0]
                print("-----")
                print(f"\n{chty}: {cur_lab[c]}")
                if significant_clusters.size == 0:
                    print("No significant clusters found")
                    for cl_idx in nonsig_clusters:
                        print(f"Non-significant cluster (p-value = {cluster_p_values[cl_idx]:.4f}):")
                else:
                    for cl_idx in significant_clusters:
                        # Display the significant channels, frequencies, and time points
                        print(f"(Marginally) significant cluster (p-value = {cluster_p_values[cl_idx]:.4f}):")
                        print(f"(Marginally) significant time points: {ref_evoked.times[clusters[cl_idx][0].start]} - {ref_evoked.times[clusters[cl_idx][0].stop]} ")
                print("-----\n")

            # save the CBP results
            clus_results = {
                            't_values'      : t_values,
                            'clusters'      : clus,
                            'cluster_ps'    : clus_ps,
                            'time_window'   : ref_evoked.times,
                            'channels'      : ref_evoked.ch_names,
                            }
            fname = os.path.join(paths["outp_MEG"],analysis['type'], 'CBP_results_ERF_'+chty+'_'+cbp_type+'.json')
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(clus_results, f, indent=2, default=to_json_safe)

            # Save X
            npz_fname = os.path.join(
                paths["outp_MEG"],
                analysis['type'],
                f"CBP_X_ERF_{chty}_{cbp_type}.npz"
            )
            np.savez_compressed(
                npz_fname,
                **{f"X_{prime}": X[prime] for prime in task['primes']},
                X_z=X['z']
            )
            meta = {
                "cbp_type": cbp_type,
                "channel_type": chty,
                "conditions": task["primes"],
                "time_window": ref_evoked.times.tolist(),
                "channels": ref_evoked.ch_names,
            }

            json_fname = os.path.join(
                paths["outp_MEG"],
                analysis['type'],
                f"CBP_X_ERF_{chty}_{cbp_type}_meta.json"
            )

            with open(json_fname, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=to_json_safe)
            
            ### PLOTTING
            
            ## CBP ERFs
            # make the topoplot showing the sensors
            if chty =='grad':
                tmp = ERF_g_ga['p0'].copy()
                t = mne.viz.plot_sensors(tmp.pick(picks=np.unique(meg['chan_RS_g']+meg['chan_RE_g'])).info, show=False)
            elif chty =='magl':
                tmp = ERF_m_ga['p0'].copy()
                t = mne.viz.plot_sensors(tmp.pick(picks=np.unique(meg['chan_RS_ml']+meg['chan_RE_ml'])).info, show=False)
            elif chty =='magr':
                tmp = ERF_m_ga['p0'].copy()
                t = mne.viz.plot_sensors(tmp.pick(picks=np.unique(meg['chan_RS_mr']+meg['chan_RE_mr'])).info, show=False)
            im = fig2img(t)
            im = im.resize(np.multiply(im.size,.8).astype(int))
            # ERF
            times = ref_evoked.times
            ax = {}; ax[0] = []
            plt.rc('font',size = 15)
            plt.rcParams['axes.grid'] = False
            seaborn.set_palette(analysis['pltcol'])
            figsz = (15, 10)
            fig = plt.figure(dpi=300)
            fig.set_figheight(figsz[0])
            fig.set_figwidth(figsz[1])
            spec = gridspec.GridSpec(ncols=1, nrows=2,
                width_ratios=[1], wspace=0,
                hspace=.7, height_ratios=[1, 1])
            ax1 = fig.add_subplot(spec[0])
            ax[0] = fig.add_subplot(spec[1])
            for prime in range(len(task['primes'])):
                plt_data = np.mean(X[task['primes'][prime]],0) # average over the first dimension (subjects)  
                # transform data from T to fT and T/m to fT/cm
                if 'grad' in chty:
                    plt_data = np.multiply(plt_data, 1e13)
                elif 'mag' in chty:
                    plt_data = np.multiply(plt_data, 1e15)
                ax1.plot(
                    times,
                    plt_data,
                    label=task['pr_lab'][prime],
                )
            if 'grad' in chty:
                title_main = "Gradiometers"
                ax1.set_ylabel("fT / cm")
                ax1.set_ylim(0,34)
            elif 'mag' in chty:
                ax1.set_ylabel("fT")
                if chty == 'magl':
                    title_main = "Magnetometers (left)"
                    ax1.set_ylim(-30,80)
                    ax1.invert_yaxis()
                    ax1.axhline(0, color='k', linestyle='--')
                elif chty == 'magr':
                    title_main = "Magnetometers (right)"
                    ax1.set_ylim(-80,30)
                    ax1.axhline(0, color='k', linestyle='--')
            ax1.set_title("ERFs")
            ax1.set_xlabel("time (s)")
            ax1.set_xlim(-.1,.5)
            ax1.axvline(0, color='k', linestyle='--')
            ax1.legend(bbox_to_anchor =(0.5,-.3), loc='lower center', ncol=3)
            # add the image
            fig_sz = fig.get_size_inches()*fig.dpi
            fig.figimage(im, xo=fig_sz[0]-im.size[0]*1.5, yo=fig_sz[1]-im.size[1]*1, zorder=10)
            # Difference
            diff_lab = ["1-back prime - no prime", "2-back prime - no prime"]
            for c2 in range(len(cur_X)):
                for i_c, c in enumerate(clus[c2]): # i_c = cluster index, c = cluster start and end
                    c = c[0]
                    if clus_ps[c2][i_c] < 0.05:
                        h  = ax[0].axvspan(times[c.start], times[c.stop - 1], color=analysis['pltcol'][1:][c2], alpha=0.3)
                    elif clus_ps[c2][i_c] < 0.1:
                        ax[0].axvline(times[c.start],color=analysis['pltcol'][1:][c2], linestyle='dotted')
                        ax[0].axvline(times[c.stop - 1],color=analysis['pltcol'][1:][c2], linestyle='dotted')
                        h  = ax[0].axvspan(times[c.start], times[c.stop - 1], color=analysis['pltcol'][1:][c2], alpha=0.3, hatch='xx', fill=False)
                    else:
                        pass
                ax[0].plot(times, t_values[c2], color=analysis['pltcol'][1:][c2], label=diff_lab[c2])
            ax[0].legend(bbox_to_anchor =(0.5,-.3), loc='lower center', ncol=2)
            ax[0].set_title('Condition Differences (t-values)', pad=figsz[1]*1.5)
            if c2 == len(cur_X)-1:
                ax[0].set_xlabel("time (s)")
            ax[0].set_ylabel("t-values")
            ax[0].set_ylim(-8.5,8.5)
            if chty == 'magl':
                ax[0].invert_yaxis()
            ax[0].set_xlim(-.1,.5)
            ax[0].axhline(0, color='k', linestyle='--')
            ax[0].axvline(0, color='k', linestyle='--')
            # main title
            fig.suptitle(title_main,  fontweight ="bold", x = .52, y=.98)
            fig
            # save the plot
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], cur_name), dpi=300)
            plt.close('all')
        # Plot all plots side by side
        images = []
        for c in range(len(cn_types)):
            if cn_types[c]=='grad':
                cur_name = 'GA_ERFs_'+cbp_type+'_grad_perm.png'
            elif cn_types[c]=='magl':
                cur_name = 'GA_ERFs_'+cbp_type+'_magl_perm.png'
            elif cn_types[c]=='magr':
                cur_name = 'GA_ERFs_'+cbp_type+'_magr_perm.png'
            images.append(Image.open(os.path.join(paths["outp_MEG"],analysis['type'], cur_name)))
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height), color="white")
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        lab_n = ("A", "C", "E", "B", "D", "F")
        lab_x = [0, widths[0], widths[0] + widths[1], 0, widths[0], widths[0] + widths[1]]
        lab_x = [x + total_width*.01 for x in lab_x] # adjust position slightly
        lab_y = (max_height*.01, max_height*.01, max_height*.01, max_height/2, max_height/2, max_height/2) # adjust position slightly
        try:
            font = ImageFont.truetype("arial.ttf", size=150)
        except IOError:
            font = ImageFont.load_default() # the font size of this cannot be changed, so it will be too tiny to be visible
        draw = ImageDraw.Draw(new_im)
        for lab in range(len(lab_n)):
            draw.text((lab_x[lab], lab_y[lab]), lab_n[lab], fill="Black", font = font, align ="left")  
        new_im.show()
        new_im.save(os.path.join(paths["outp_MEG"],analysis['type'], 'GA_ERFs_cbp_type_all.png'), dpi=(300, 300))

        if analysis['erf']['vep_calc']:
            ## FIND THE CHANNELS WITH THE BIGGEST VEP
            # https://www.sciencedirect.com/science/article/pii/B9780444640321000345#f0035
            # gradiometers
            # get the peak max(abs()) values per channel for the timepoints between 0 and 500 ms
            tms     = [t>0 and t<=.5 for t in ERF_g_vep_ga.times] 
            absmax  = np.max(np.abs(ERF_g_vep_ga.data[:,tms]),axis=1)
            # and choose the 10 channels with the higest amplitude
            maxREchns_g = [ERF_g_vep_ga.ch_names[i] for i in np.flip(np.argsort(absmax))][0:10]
            # magnetometers
            # get the peak min() and max() values per channel for the timepoints between 0 and 500 ms
            tms     = [t>0 and t<=.5 for t in ERF_m_vep_ga.times] 
            magmax  = np.max(ERF_m_vep_ga.data[:,tms],axis=1)
            magmin  = np.min(ERF_m_vep_ga.data[:,tms],axis=1)
            # and choose the 10 channels with the higest or lowest amplitude
            maxREchns_m = [ERF_m_vep_ga.ch_names[i] for i in np.flip(np.argsort(magmax))][0:10]
            minREchns_m = [ERF_m_vep_ga.ch_names[i] for i in np.argsort(magmin)][0:10]
            
            # PLOT RS and RE
            # adjust units as MNE seems confused
            ERF_g_vep_ga._data = np.multiply(ERF_g_vep_ga._data, 1e-2)
            # https://mne.tools/stable/generated/mne.viz.plot_epochs_image.html#mne.viz.plot_epochs_image
            # plot the grand averages (all)
            fig = ERF_g_vep_ga.plot(picks=np.unique(meg['chan_RS_g']+meg['chan_RE_g']), gfp=True, titles=dict(mag='RS & RE (grad)'))
            fig.set_size_inches((20, 10), forward=True) 
            fig = ERF_m_vep_ga.plot(picks=np.unique(meg['chan_RS_ml']+meg['chan_RE_ml']), gfp=True, titles=dict(mag='RS & RE (mag left)'))
            fig.set_size_inches((20, 10), forward=True) 
            fig = ERF_m_vep_ga.plot(picks=np.unique(meg['chan_RS_mr']+meg['chan_RE_mr']), gfp=True, titles=dict(mag='RS & RE (mag right)'))
            fig.set_size_inches((20, 10), forward=True) 
            fig = ERF_g_vep_ga.plot_topomap(times=[0.15, 0.4], average=[0.1, 0.2], sensors = False, show_names=True)
            fig.set_size_inches((30, 15), forward=True)
            fig = ERF_m_vep_ga.plot_topomap(times=[0.15, 0.4], average=[0.1, 0.2], sensors = False, show_names=True)
            fig.set_size_inches((30, 15), forward=True)
            # create masks
            times       = ERF_g_vep_ga.times
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=5)
            mask_g_chns = np.isin(ERF_g_vep_ga.ch_names, meg['chan_RS_g']+meg['chan_RE_g'])
            mask_g      = np.tile(mask_g_chns.reshape(len(mask_g_chns),1), (1, len(times)))
            mask_m_chns = np.isin(ERF_m_vep_ga.ch_names, meg['chan_RS_ml']+meg['chan_RE_ml']+meg['chan_RS_mr']+meg['chan_RE_mr'])
            mask_m      = np.tile(mask_m_chns.reshape(len(mask_m_chns),1), (1, len(times)))

            fig = ERF_g_vep_ga.plot_topomap(times=0.25, average=[0.5], sensors = False, mask = mask_g, mask_params = mask_params, vlim = (0, 21), cbar_fmt="%d", time_format = "", units = "")
            fig.set_size_inches((5, 5), forward=True)
            # save plot
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], 'GA_VEPs_grad.png'), bbox_inches='tight', dpi=300)
            
            fig = ERF_m_vep_ga.plot_topomap(times=0.25, average=[0.5], sensors = False, mask = mask_m, mask_params = mask_params, vlim = (-45, 45), cbar_fmt="%d", time_format = "", units = "")
            fig.set_size_inches((5, 5), forward=True)
            # save plot
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], 'GA_VEPs_mag.png'), bbox_inches='tight', dpi=300)

            # saving the images
            save_multi_image(os.path.join(paths["outp_MEG"], analysis['type'], 'GA_VEPs.pdf'))

            # combining figures
            images = [
                Image.open(os.path.join(paths["outp_MEG"],analysis['type'], 'GA_VEPs_mag.png')),
                Image.open(os.path.join(paths["outp_MEG"],analysis['type'], 'GA_VEPs_grad.png'))
            ]
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            x_offset = 0
            y_offset = 110
            new_im = Image.new('RGB', (total_width + x_offset, max_height + y_offset), color="white")
            for im in images:
                new_im.paste(im, (x_offset,y_offset))
                x_offset += im.size[0]
            draw = ImageDraw.Draw(new_im)
            lab_n = ("fT", "fT/cm")
            lab_x = (widths[0]*.82, widths[0] + widths[1]*.82)
            lab_y = (y_offset*.8, y_offset*.8)
            # Try loading the specified font, otherwise load a default font
            try:
                font = ImageFont.truetype("arial.ttf", size=40)
            except IOError:
                font = ImageFont.load_default() # will plot too small as we cannot set the font size..
            for lab in range(len(lab_n)):
                draw.text((lab_x[lab], lab_y[lab]), lab_n[lab], fill="Black", font = font, align ="center")
            
            lab_n = ("Magnetometers", "Gradiometers")
            lab_x = (widths[0]/4, widths[0] + widths[1]/4)
            lab_y = (max_height*.01, max_height*.01)
            # Try loading the specified font, otherwise load a default font
            try:
                font = ImageFont.truetype("arial.ttf", size=50)
            except IOError:
                font = ImageFont.load_default() # will plot too small as we cannot set the font size..
            draw = ImageDraw.Draw(new_im)
            for lab in range(len(lab_n)):
                draw.text((lab_x[lab], lab_y[lab]), lab_n[lab], fill="Black", font = font, align ="center")
            # save image
            new_im.save(os.path.join(paths["outp_MEG"],analysis['type'], 'GA_VEPs_all.png'), dpi=(300, 300))
            plt.close('all')
        
        ## Behav & ERF corr
        print('\n*** Memory & ERF Correlations ***\n')
        ppns_incl = []
        subj_behav_erf = []
        # loop over participantsto get the memory performance
        for subj['nr'] in sorted(os.listdir(paths['inp_MEG'])):
            #%% SAVED VARIABLES LOCATIONS
            var_loc.update({
                'sub'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json') # where the subject data is saved
                })
            with open(var_loc["sub"], "r", encoding="utf-8") as fp:
                subj = json.load(fp)

            if not subj['bad']:
                print(subj['nr'])
                ppns_incl.append(subj['nr'])
                # the memory performance for the correlation
                mem_vars = ['Memory_dprime_p0b', 'Memory_dprime_p1b', 'Memory_dprime_p2b'] 
                behav_erf_dict = {'ppn' : subj['nr']}
                behav_erf_dict.update({key: subj['Behav'][key] for key in mem_vars if key in subj['Behav']})
                behav_erf_dict.update({'ERF': subj['ERF']})
                subj_behav_erf.append(behav_erf_dict)

        # Define priming condition pairs to compare
        prime_pairs = [('p1', 'p0'), ('p2', 'p0')]

        # Store correlation results
        correlation_results = {}

        for key in t_win.keys():
            correlation_results[key] = {}
            for chty in cn_types:
                correlation_results[key][chty] = {}
                for prime_a, prime_b in prime_pairs:
                    # Collect differences across subjects
                    mem_diffs = []
                    erf_diffs = []
                    for subj in subj_behav_erf:
                        mem_key_a = f"Memory_dprime_{prime_a}b"
                        mem_key_b = f"Memory_dprime_{prime_b}b"
                        if mem_key_a in subj and mem_key_b in subj:
                            mem_diff = subj[mem_key_a] - subj[mem_key_b]
                            erf_diff = subj['ERF'][prime_a][key][chty] - subj['ERF'][prime_b][key][chty]
                            mem_diffs.append(mem_diff)
                            erf_diffs.append(erf_diff)
                    # Compute correlation
                    if len(mem_diffs) >= 2:
                        r, p = spearmanr(mem_diffs, erf_diffs)
                        correlation_results[key][chty][f"{prime_a}-{prime_b}"] = {'r': r, 'p': p}
                    else:
                        correlation_results[key][chty][f"{prime_a}-{prime_b}"] = {'r': None, 'p': None}
                    if p < .05:                    
                        plt.figure()
                        plt.scatter(mem_diffs, erf_diffs)
                        plt.title(f'Correlation\nr = {r:.2f}, p = {p:.3f}')
                        plt.xlabel('Memory Difference')
                        plt.xlim(-max(abs(min(mem_diffs)), abs(max(mem_diffs)))*1.1, max(abs(min(mem_diffs)), abs(max(mem_diffs)))*1.1)
                        plt.ylabel('ERF Difference')
                        plt.ylim(-max(abs(min(erf_diffs)), abs(max(erf_diffs)))*1.1, max(abs(min(erf_diffs)), abs(max(erf_diffs)))*1.1)
                        plt.grid(False)
                        plt.tight_layout()
                        plt.show()

        # Print results
        for key in correlation_results:
            for chty in correlation_results[key]:
                for pair in correlation_results[key][chty]:
                    result = correlation_results[key][chty][pair]
                    print(f"Time window: {key}, Channel type: {chty}, Comparison: {pair} => r = {result['r']}, p = {result['p']}")
                    
    if analysis['types']['tfr'] and analysis['type']=='ERFTFR':
        #%% Time-frequency representations of power %%
        print('\nGRAND AVERAGES: TFR ANALYSIS\n')
        var_loc = {
            'gat' : os.path.join(paths["outp_MEG"],analysis['type'],'TFR_ga_{cond}.fif'), # where the GA TFR data is saved
            'gal' : os.path.join(paths["outp_MEG"],analysis['type'],'TFR_all_{cond}_{subj}.fif'), # where the all subject data is saved
            }
         # check if the grand average data can just be loaded in
        conds = task['primes'] # conditions of interest
        subj_names = []
        for s in sorted(os.listdir(paths['inp_MEG'])):
            sub_file = os.path.join(paths["outp_MEG"], analysis['type'], f"{s}_sub.json")
            if os.path.isfile(sub_file):
                with open(sub_file) as f:
                    info = json.load(f)
                if not info.get("bad", False):
                    subj_names.append(s)
        file_paths = []
        file_paths += [var_loc['gat'].format(cond=cond) for cond in conds]
        file_paths += [var_loc['gal'].format(cond=cond, subj=subj_name) for cond in conds for subj_name in subj_names]
        cur_redo = False # True for recalculate, False for load in old data
        if all(os.path.isfile(path) for path in file_paths) and not cur_redo:
            tfr_all = {}
            tfr_ga = {}
            for cond in conds:
                tfr_ga[cond] = mne.time_frequency.read_tfrs(var_loc['gat'].format(cond=cond))[0]
                tfr_all[cond] = []
                for subj_name in subj_names: 
                    tfr_all[cond].append((
                        subj_name,
                        mne.time_frequency.read_tfrs(var_loc['gal'].format(cond=cond, subj=subj_name))[0]
                    ))
        else:
            tfr_all = {}
            tfr_ga = {}
            tfr_sub_g = {}
            tfr_sub_m = {}
            tfr_ga_g_m = list(np.zeros(len(task['primes'])))
            tfr_ga_ml_m = list(np.zeros(len(task['primes'])))
            tfr_ga_mr_m = list(np.zeros(len(task['primes'])))
            tfr_ga_g_s = list(np.zeros(len(task['primes'])))
            tfr_ga_ml_s = list(np.zeros(len(task['primes'])))
            tfr_ga_mr_s = list(np.zeros(len(task['primes'])))
            cn_types = ['magl', 'magr', 'grad']
            for prime in task['primes']:
                tfr_all[prime]=[]
                tfr_ga[prime]=[]
            t_win = [0.08, 0.3] # gamma effect
            f_win = [36, 56]
            # loop over participants
            for subj['nr'] in sorted(os.listdir(paths['inp_MEG'])):
                tfr = {}
                # SAVED VARIABLES LOCATIONS
                var_loc.update({
                            'tfr' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_tfr_{cond}.fif'), # where the TFR data is saved
                            'sub' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json'), # where the subject data is saved
                        })
                # get the subject info      
                with open(var_loc["sub"], "r", encoding="utf-8") as fp:
                    subj = json.load(fp)

                # only add the subject to the grand average if it is not marked as bad
                if not subj['bad']:
                    subj['TFR'] = {}
                    # loop over conditions
                    for prime in task['primes']:
                        # load in the data per participant
                        tfr[prime] = mne.time_frequency.read_tfrs(var_loc['tfr'].format(cond=prime))[0]
                        tfr_sub = (
                            tfr[prime]
                            .copy()
                            .pick(mne.pick_channels_regexp(tfr[prime].info['ch_names'],'MEG*'))
                            .crop(tmin=-1, tmax=0.61)
                        )
                        tfr_all[prime].append((subj['nr'], tfr_sub))
                        subj['TFR'][prime] = {}
                        # Save the data for the correlation with behavior
                        cur_tms_mask = (tfr[prime].times > t_win[0] - 0.001) & (tfr[prime].times < t_win[1] + 0.001)
                        cur_frq_mask = (tfr[prime].freqs > f_win[0] - 0.001) & (tfr[prime].freqs < f_win[1] + 0.001)
                        cur_data = tfr[prime].data
                        for chty in cn_types:
                            if chty=='grad':
                                cur_chns_mask = np.isin(tfr[prime].ch_names, meg['chan_RS_gp']+meg['chan_RE_gp'])
                                selected_data = cur_data[np.ix_(cur_chns_mask, cur_frq_mask, cur_tms_mask)]
                                val = selected_data.mean()
                            elif chty=='magl':
                                cur_chns_mask = np.isin(tfr[prime].ch_names, meg['chan_RS_ml']+meg['chan_RE_ml'])
                                selected_data = cur_data[np.ix_(cur_chns_mask, cur_frq_mask, cur_tms_mask)]
                                val = selected_data.mean()
                            elif chty=='magr':
                                cur_chns_mask = np.isin(tfr[prime].ch_names, meg['chan_RS_mr']+meg['chan_RE_mr'])
                                selected_data = cur_data[np.ix_(cur_chns_mask, cur_frq_mask, cur_tms_mask)]
                                val = selected_data.mean()
                            subj['TFR'][prime][chty] = val

                    # save the subject data      
                    with open(var_loc["sub"], "w", encoding="utf-8") as fp:
                        json.dump(subj, fp, indent=2, default=to_json_safe)

            # make the grand averages
            for prime in task['primes']:
                tfr_ga[prime] = mne.grand_average([tfr for _, tfr in tfr_all[prime]])
                tfr_ga[prime].comment = prime + ' ' + tfr_ga[prime].comment
            
            print('saving files')
            for prime, tfr in tfr_ga.items():
                tfr.save(
                    var_loc['gat'].format(cond=prime),
                    overwrite=True
                )
            for prime, pairs in tfr_all.items():
                for subj_name, tfr in pairs:
                    tfr.save(
                        var_loc['gal'].format(cond=prime, subj=subj_name),
                        overwrite=True
                    )
        
        # CLUSTER-BASED PERMUTATION
        # https://mne.tools/stable/auto_examples/stats/cluster_stats_evoked.html#sphx-glr-auto-examples-stats-cluster-stats-evoked-py
        # https://mne.tools/stable/auto_tutorials/stats-sensor-space/40_cluster_1samp_time_freq.html#sphx-glr-auto-tutorials-stats-sensor-space-40-cluster-1samp-time-freq-py
        # https://mne.discourse.group/t/how-to-set-up-a-cluster-permutation-test-between-two-independent-sample-groups/4668/2
        # loop over the various channel types (3)
        
        for chty in cn_types:
            # reshape the data for the CBP
            X = {};
            sel = {};
            for prime in task['primes']:
                X[prime]=[]
                sel[prime]=[]
                cur_data = tfr_all[prime]
                cur_frqs = analysis['tfr']['gamma_freqs']
                cur_times = analysis['tfr']['cbp_times']
                if chty=='grad':
                    cur_chns = np.unique(meg['chan_RS_gp']+meg['chan_RE_gp']); 
                    cur_chtyp = 'grad'
                    cur_name = 'GA_TFRs_grad_perm.png'
                elif chty=='magl':
                    cur_chns = np.unique(meg['chan_RS_ml']+meg['chan_RE_ml']); 
                    cur_chtyp = 'mag'
                    cur_name = 'GA_TFRs_magl_perm.png'
                elif chty=='magr':
                    cur_chns = np.unique(meg['chan_RS_mr']+meg['chan_RE_mr']); 
                    cur_chtyp = 'mag'
                    cur_name = 'GA_TFRs_magr_perm.png'
                # extract a reference TFR once
                _, ref_tfr = cur_data[0]
                for _, tfr in cur_data:  # Each TFR is for one subject
                    tmp = tfr.data  # Subject-specific TFR data
                    cns = np.isin(tfr.ch_names, cur_chns)  # Select relevant channels
                    fqs = np.isin(tfr.freqs, cur_frqs)  # Select relevant frequencies
                    tmp = tmp[cns, :][:, fqs, :][:, :, :]  # Apply selections (n_channels, n_frequencies, n_times)
                    tmp = np.nanmean(tmp,axis=0) # average over channels (n_frequencies, n_times)
                    X[prime].append(tmp)
                # Stack subject data into a consistent shape
                X[prime] = np.stack(X[prime])  # Shape: (n_subjects, n_frequencies, n_times)
                # only test the data in the relevant time window
                tms = (ref_tfr.times<cur_times[0]-.001) | (ref_tfr.times>cur_times[1]+.001)
                tms_ext = np.expand_dims(tms, axis=(0))
                sel[prime] = np.broadcast_to(tms_ext, X[prime][0].shape) # selection of relevant time window (inverted)

            # channel adjacency
            # get the existing adjacency matrix
            sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(ref_tfr.info, cur_chtyp)
            # subselect channels we use
            ch_names2 = [x.replace(" ","") for x in ch_names] # remove the space to compare the lists
            use_idx = np.isin(ch_names2,cur_chns)
            sensor_adjacency = sensor_adjacency[use_idx][:, use_idx]
            # temporal adjacency (Gaussian)
            temporal_adjacency = gaussian_adjacency_binary(n = X['p0'].shape[2], sigma = 8) 
            # frequency adjacency (Gaussian)
            frequency_adjacency = gaussian_adjacency_binary(n = X['p0'].shape[1], sigma = 3) 
            # prepare the adjacency of the time-frequency plane

            adjacency = mne.stats.combine_adjacency(frequency_adjacency, temporal_adjacency)  # Time-frequency adjacency

            # continue preparing the actual cluster-based permutation
            X['z']=np.zeros(np.shape(X['p0'])) # to potentially test against 0
            cur_lab = ['1-back prime - no prime', '2-back prime - no prime']
            cur_X = []
            cur_X = [
                (np.array(X['p1']) - np.array(X['p0'])) / (np.array(X['p1']) + np.array(X['p0'])),  # Comparison: p1 vs. p0
                (np.array(X['p2']) - np.array(X['p0'])) / (np.array(X['p2']) + np.array(X['p0'])),  # Comparison: p2 vs. p0
            ]
            print("---")
            print("(A-B)/(A+B)")
            print("---")
            cur_sel = sel['p0'] # we can just use the one from the first condition as they are all the same
            # Run permutation cluster tests
            t_values=[]
            clus=[]
            clus_ps=[]
            for c in range(len(cur_X)):
                t_thresh = st.t.ppf(1 - analysis['tfr']['cbp_clalpha'] / 2, df=X['z'].shape[0]-1) # cluster p = .005 (two-tailed)
                print(f"t_thresh = {round(t_thresh,3)}")
                t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                    X               = cur_X[c],     # Data for comparison
                    out_type        = "mask",       # boolean arrays where true indicates part of a cluster
                    n_permutations  = analysis['tfr']['cbp_nperm'],         # number of permutations to compute
                    threshold       = t_thresh,
                    tail            = 0,
                    seed            = 0,
                    exclude         = cur_sel.flatten(),
                    adjacency       = adjacency,
                )
                # save the data in list
                t_values.append(t_obs)
                clus.append(clusters)
                clus_ps.append(cluster_p_values)
                # Print significant clusters
                significant_clusters = np.where(cluster_p_values < 0.05)[0]
                nonsig_clusters = np.where(cluster_p_values >= 0.05)[0]
                print("-----")
                print(f"\n{chty}: {cur_lab[c]}")
                if significant_clusters.size == 0:
                    print("No significant clusters found.")
                    for cl_idx in nonsig_clusters:
                        print(f"Non-significant cluster (p-value = {cluster_p_values[cl_idx]:.4f}):")
                else:
                    for cl_idx in significant_clusters:
                        # Find the significant time points by collapsing over channels and frequencies
                        sig_times = np.where(np.any(clusters[cl_idx], axis=(0)))[0]  # Collapse over channels and frequencies
                        # Find the significant channels and frequencies
                        sig_frequencies = np.where(np.any(clusters[cl_idx], axis=(1)))[0]  # Collapse over channels and time
                        # Extract corresponding channel names and frequency values
                        cur_tms = ref_tfr.times
                        significant_times = cur_tms[sig_times]  # Extract specific time points from `cur_times`
                        frequency_values = [cur_frqs[i] for i in sig_frequencies]  # Replace `cur_frqs` with your frequency list
                        # Display the significant channels, frequencies, and time points
                        print(f"Significant cluster (p-value = {cluster_p_values[cl_idx]:.4f}):")
                        print(f"Significant time points: {significant_times}")
                        # print(f"Significant channels: {channel_names}")
                        print(f"Significant frequencies: {frequency_values}")
                print("-----\n")
            # save the results
            clus_results = {
                            't_values'      : t_values,
                            'clusters'      : clus,
                            'cluster_ps'    : clus_ps,
                            'time_window'   : np.tile(cur_times,(len(cur_X),1)),
                            'channels'      : np.tile(cur_chns,(len(cur_X),1)),
                            'freqs'         : np.tile(cur_frqs,(len(cur_X),1)),
                            }
            fname = os.path.join(paths["outp_MEG"],analysis['type'], 'CBP_results_TFR_'+chty+'_'+cbp_type+'.json')
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(clus_results, f, indent=2, default=to_json_safe)
            
            # Save X
            npz_fname = os.path.join(
                paths["outp_MEG"],
                analysis['type'],
                f"CBP_X_TFR_{chty}_{cbp_type}.npz"
            )
            np.savez_compressed(
                npz_fname,
                **{f"X_{prime}": X[prime] for prime in task['primes']},
                X_z=X['z']
            )
            meta = {
                "cbp_type": cbp_type,
                "channel_type": chty,
                "conditions": task["primes"],
                "time_window": ref_tfr.times.tolist(),
                "channels": ref_tfr.ch_names,
            }

            json_fname = os.path.join(
                paths["outp_MEG"],
                analysis['type'],
                f"CBP_X_TFR_{chty}_{cbp_type}_meta.json"
            )

            with open(json_fname, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=to_json_safe)

            # PLOTTING
            # make the topoplot showing the sensors
            if chty =='grad':
                title_main = "Gradiometers"
                tmp = tfr_ga['p0'].copy()
                t = mne.viz.plot_sensors(tmp.pick(picks=np.unique(meg['chan_RS_g']+meg['chan_RE_g'])).info)
            elif chty =='magl':
                title_main = "Magnetometers (left)"
                tmp = tfr_ga['p0'].copy()
                t = mne.viz.plot_sensors(tmp.pick(picks=np.unique(meg['chan_RS_ml']+meg['chan_RE_ml'])).info)
            elif chty =='magr':
                title_main = "Magnetometers (right)"
                tmp = tfr_ga['p0'].copy()
                t = mne.viz.plot_sensors(tmp.pick(picks=np.unique(meg['chan_RS_mr']+meg['chan_RE_mr'])).info)
            topo_im = fig2img(t)
            topo_im = topo_im.resize(np.multiply(topo_im.size,.8).astype(int))
            # make the TFRs
            times = analysis['tfr']['cbp_times']
            freqs = analysis['tfr']['gamma_freqs']
            plt.rcParams['font.size'] = 15 
            seaborn.set_palette(analysis['pltcol'])
            figsz = (15, 10)
            fig = plt.figure(dpi=300)
            fig.set_figheight(figsz[0])
            fig.set_figwidth(figsz[1])
            ax = {}
            spec = gridspec.GridSpec(ncols=1, nrows=2,
                width_ratios=[1], wspace=0,
                hspace=0.7, height_ratios=[1, 1.3])
            ax[0] = fig.add_subplot(spec[0])
            ax[1] = fig.add_subplot(spec[1])
            for c2 in range(2):
                cur_t_values    = t_values[c2]
                cur_clus        = clus[c2]
                mar = False # marginal
                sig = False # significant
                for c, p_val in zip(cur_clus, clus_ps[c2]):
                    if p_val <= .05: # only plot the sig effects
                        # get the significant timepoints
                        mar = True
                        sig_times_idx = np.where(np.any(c, axis=(0)))[0]
                        times_all = ref_tfr.times
                        sig_times = times_all[sig_times_idx]
                        # for inclusive plotting                    
                        if sig_times.size > 0:
                            dt = np.median(np.diff(times_all)) if times_all.size > 1 else 0.0
                            sig_times = np.append(sig_times, sig_times[-1] + dt)

                        sig_freqs_idx = np.where(np.any(c, axis=(1)))[0]
                        freqs_all = np.asarray(freqs)
                        sig_freqs = freqs_all[sig_freqs_idx]
                        # for inclusive plotting                      
                        if sig_freqs.size > 0:
                            df = np.median(np.diff(freqs_all)) if freqs_all.size > 1 else 0.0
                            sig_freqs = np.append(sig_freqs, sig_freqs[-1] + df)

                        if p_val < .05:
                            sig = True
                            mar = False
                # select time window
                t_win = (-.1, .5)
                cur_t_values = cur_t_values[:,(ref_tfr.times>t_win[0]-.001) & (ref_tfr.times<t_win[1]+.001)]
                vmax = st.t.ppf(1 - analysis['tfr']['cbp_clalpha'] / 2, df=X['p0'].shape[0]) # t-value at p = .01 (two-tailed)
                vmin = -vmax
                plt_tfr = ax[c2].imshow(
                    cur_t_values,
                    cmap=plt.cm.RdBu_r,
                    extent=[t_win[0], t_win[-1], freqs[0], freqs[-1]],
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[c2].axvline(0, color='k', linestyle='--')

                if mar:
                    # Add the box
                    rect = Rectangle(
                        (sig_times[0], sig_freqs[0]), sig_times[-1]-sig_times[0], sig_freqs[-1]-sig_freqs[0],  # Box position and size
                        linewidth=4, edgecolor='white', facecolor='none', linestyle='--'
                    )
                    ax[c2].add_patch(rect)
                if sig:
                    # Add the box
                    rect = Rectangle(
                        (sig_times[0], sig_freqs[0]), sig_times[-1]-sig_times[0], sig_freqs[-1]-sig_freqs[0],  # Box position and size
                        linewidth=4, edgecolor='white', facecolor='none', linestyle='-'
                    )
                    ax[c2].add_patch(rect)

                ax[c2].set_xlabel("Time (s)")
                ax[c2].set_ylabel("Frequency (Hz)")
                ax[c2].set_title(f"{cur_lab[c2]}", pad=figsz[1]*1.5)
            # add colorbar
            fig.colorbar(plt_tfr, ax=ax[c2], location='bottom', fraction = 0.05, aspect = 60, pad = .20)
            # main title
            fig.suptitle(title_main,  fontweight ="bold", x = .51, y=.98) 
            # add the topoplot
            fig_sz = fig.get_size_inches()*fig.dpi
            fig.figimage(
                topo_im, 
                xo=fig_sz[0]-topo_im.size[0]*1.5, 
                yo=fig_sz[1]-topo_im.size[1]*1, 
                zorder=10)
            fig
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], cur_name), dpi=300)
            plt.close('all')

        # Plot all plots side by side
        images = []
        for c in range(len(cn_types)):
            if cn_types[c]=='grad':
                cur_name = 'GA_TFRs_grad_perm.png'
            elif cn_types[c]=='magl':
                cur_name = 'GA_TFRs_magl_perm.png'
            elif cn_types[c]=='magr':
                cur_name = 'GA_TFRs_magr_perm.png'
            images.append(Image.open(os.path.join(paths["outp_MEG"],analysis['type'], cur_name)))
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height-round(max_height*0.07)), color="white")
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        lab_n = ("A", "C", "E", "B", "D", "F")
        lab_x = [0, widths[0], widths[0] + widths[1], 0, widths[0], widths[0] + widths[1]]
        lab_x = [x + total_width*.01 for x in lab_x] # adjust position slightly
        lab_y = (max_height*.01, max_height*.01, max_height*.01, max_height/2, max_height/2, max_height/2) # adjust position slightly
        try:
            font = ImageFont.truetype("arial.ttf", size=150)
        except IOError:
            font = ImageFont.load_default() # the font size of this cannot be changed, so it will be too tiny to be visible
        draw = ImageDraw.Draw(new_im)
        for lab in range(len(lab_n)):
            draw.text((lab_x[lab], lab_y[lab]), lab_n[lab], fill="Black", font = font, align ="left")  
        new_im.save(os.path.join(paths["outp_MEG"],analysis['type'], 'GA_TFRs_all_perm.png'),dpi=(300, 300))


        ## Behav & TFR corr
        print('\n*** Memory & TFR Correlations ***\n')
        ppns_incl = []
        subj_behav_tfr = []
        # loop over participantsto get the memory performance
        for subj['nr'] in sorted(os.listdir(paths['inp_MEG'])):
            #%% SAVED VARIABLES LOCATIONS
            var_loc.update({
                'sub'       : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json') # where the subject data is saved
                })     
            with open(var_loc["sub"], "r", encoding="utf-8") as fp:
                subj = json.load(fp)

            if not subj['bad']:
                print(subj['nr'])
                ppns_incl.append(subj['nr'])
                # the memory performance for the correlation
                mem_vars = ['Memory_dprime_p0b', 'Memory_dprime_p1b', 'Memory_dprime_p2b'] 
                behav_tfr_dict = {'ppn' : subj['nr']}
                behav_tfr_dict.update({key: subj['Behav'][key] for key in mem_vars if key in subj['Behav']})
                behav_tfr_dict.update({'TFR': subj['TFR']})
                subj_behav_tfr.append(behav_tfr_dict)

        # Define priming condition pairs to compare
        prime_pairs = [('p1', 'p0'), ('p2', 'p0')]

        # Store correlation results
        correlation_results = {}

        for chty in cn_types:
            correlation_results[chty] = {}
            for prime_a, prime_b in prime_pairs:
                # Collect differences across subjects
                mem_diffs = []
                tfr_diffs = []
                for subj in subj_behav_tfr:
                    mem_a = f"Memory_dprime_{prime_a}b"
                    mem_b = f"Memory_dprime_{prime_b}b"
                    if mem_a in subj and mem_b in subj:
                        mem_diff = subj[mem_a] - subj[mem_b]
                        tfr_diff = subj['TFR'][prime_a][chty] - subj['TFR'][prime_b][chty]
                        mem_diffs.append(mem_diff)
                        tfr_diffs.append(tfr_diff)
                if len(mem_diffs) >= 2:
                    r, p = spearmanr(mem_diffs, tfr_diffs)
                    correlation_results[chty][f"{prime_a}-{prime_b}"] = {'r': r, 'p': p}
                else:
                    correlation_results[chty][f"{prime_a}-{prime_b}"] = {'r': None, 'p': None}
                if p < .05:                    
                    plt.figure()
                    plt.scatter(mem_diffs, tfr_diffs)
                    plt.title(f'Correlation\nr = {r:.2f}, p = {p:.3f}')
                    plt.xlabel('Memory Difference')
                    plt.xlim(-max(abs(min(mem_diffs)), abs(max(mem_diffs)))*1.1, max(abs(min(mem_diffs)), abs(max(mem_diffs)))*1.1)
                    plt.ylabel('TFR Difference')
                    plt.ylim(-max(abs(min(tfr_diffs)), abs(max(tfr_diffs)))*1.1, max(abs(min(tfr_diffs)), abs(max(tfr_diffs)))*1.1)
                    plt.grid(False)
                    plt.tight_layout()
                    plt.show()

        # Print results
        for chty in correlation_results:
            for pair in correlation_results[chty]:
                result = correlation_results[chty][pair]
                print(f"Time window: Channel type: {chty}, Comparison: {pair} => r = {result['r']}, p = {result['p']}")

    if analysis['types']['mvpa'] and analysis['type']=='SVM':
        #%% Classification (SVM)
        print('\nGRAND AVERAGES: CLASSIFICATION\n')
        svm_scores = []
        tmp = []
        all_subjects = []
        ppns_incl = []
        subj_behav_mem = []
        # loop over participants
        for subj['nr'] in sorted(os.listdir(paths['inp_MEG'])):
            #%% SAVED VARIABLES LOCATIONS
            var_loc = {
                'svm' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_svm_{prime}_{cat}.nc'), # where the SVM data is saved
                'sub' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json') # where the subject data is saved
            }
            with open(var_loc["sub"], "r", encoding="utf-8") as fp:
                subj = json.load(fp)
            subj["SVM_time"] = np.asarray(subj["SVM_time"], dtype=float)
            # find all SVM files for this subject
            svm_files = glob(var_loc["svm"].format(prime="*", cat="*"))

            # only add the subject to the grand average if it is available and not marked as bad
            if not subj['bad'] and len(svm_files) != 0:
                print(subj['nr'])
                ppns_incl.append(subj['nr'])
                # load in the data for this participant
                das = []     
                das_by_prime = {}
                for f in svm_files:
                    da = xr.load_dataarray(f)
                    prime = da.coords["prime"].item()
                    cat   = da.coords["cat"].item()
                    da = da.expand_dims(
                        subject=[subj['nr']],
                        cat=[cat],
                    )        
                    if prime not in das_by_prime:
                        das_by_prime[prime] = []
                    das_by_prime[prime].append(da)   
                # concat cats within each prime
                da_primes = []
                for prime, da_list in das_by_prime.items():
                    da_p = xr.concat(da_list, dim="cat")
                    da_p = da_p.expand_dims(prime=[prime])
                    da_primes.append(da_p)
                # concat primes
                da_subj = xr.concat(da_primes, dim="prime")
                da_subj = da_subj.transpose("subject", "prime", "cat", "time")
                all_subjects.append(da_subj)

                # score the memory performance for the correlation
                mem_vars = ['Memory_dprime_p0b', 'Memory_dprime_p1b', 'Memory_dprime_p2b']
                mem_dict = {'ppn' : subj['nr']}
                mem_dict.update({key: subj['Behav'][key] for key in mem_vars if key in subj['Behav']})
                subj_behav_mem.append(mem_dict)
            
        # combine all subjects
        svm_scores = xr.concat(all_subjects, dim="subject")

        #%% SET_UP FOR THE SPECIFIC (CATEGORY vs OTHERS) SVM CLASSIFIER
        svm_scores_ga       = {}
        svm_scores_ga_std   = {}
        svm_scores_ga_sem   = {}
        svm_scores_ga_ci    = {}
        #for prime in task['primes']:
        for prime in svm_scores.coords["prime"].values:
            # compute the averages over participants
            # select prime
            da_p = svm_scores.sel(prime=prime)
            # optionally average over categories first
            da_p = da_p.mean(dim="cat")
            # number of subjects
            n_subj = da_p.sizes["subject"]
            # mean over subjects
            mu = da_p.mean(dim="subject").values
            # std over subjects
            sd = da_p.std(dim="subject", ddof=1).values
            # SEM over subjects
            se = sd / np.sqrt(n_subj)
            # 95% CI
            ci_low, ci_high = st.t.interval(
                confidence=0.95,
                df=n_subj - 1,
                loc=mu,
                scale=se,
            )
            # store results
            svm_scores_ga[prime]     = mu
            svm_scores_ga_std[prime] = sd
            svm_scores_ga_sem[prime] = se
            svm_scores_ga_ci[prime]  = (ci_low, ci_high)
    
                                     
        # CLUSTER-BASED PERMUTATION
        # https://mne.tools/stable/auto_examples/stats/cluster_stats_evoked.html#sphx-glr-auto-examples-stats-cluster-stats-evoked-py
        # https://mne.tools/stable/auto_tutorials/stats-sensor-space/40_cluster_1samp_time_freq.html#sphx-glr-auto-tutorials-stats-sensor-space-40-cluster-1samp-time-freq-py
        # https://mne.discourse.group/t/how-to-set-up-a-cluster-permutation-test-between-two-independent-sample-groups/4668/2
        # https://mne.tools/stable/auto_tutorials/stats-sensor-space/10_background_stats.html#tfce-example
        # reshape the data for the CBP
        combs = [task['primes']]
        cur_times = analysis['mvpa']['cbp_times']
        clus_results = {}
        for c in combs:
            X = {}
            sel = {}
            if c == ['p0', 'p1', 'p2']:
                c_cat = 'all'
            else:
                c_cat = c[0][-3:]
            print('\n'+c_cat+'\n')
            cur_name = 'GA_SVM_perm_'+c_cat+'.png'
            for p in range(len(c)):
                X[c[p]]=[]
                sel[c[p]]=[]
                X[c[p]] = (
                    svm_scores
                    .sel(prime=c[p])
                    .mean(dim="cat") # average over categories
                    .transpose("subject", "time")
                    .values # convert to numpy
                )
                # only test the data in the relevant time window (0 - 500 ms)
                sel[c[p]] = (subj['SVM_time']<cur_times[0]-.001) | (subj['SVM_time']>cur_times[1]+.001) # selection of relevant time window
            # start running the actual cluster-based permutation
            if c == ['p0', 'p1', 'p2']:
                X['z']=np.zeros(np.shape(X['p1'])) # to potentially test against 0
                cur_X = [X['p1']-X['p0'], X['p2']-X['p0']]
                cur_sel = sel['p0'] # we can just use the one from the first condition as they are all the same
            else:
                X['z']=np.zeros(np.shape(X['p1_'+c_cat])) # to potentially test against 0
                cur_X = [X['p1_'+c_cat]-X['p0_'+c_cat], X['p2_'+c_cat]-X['p0_'+c_cat]]
                cur_sel = sel['p0_'+c_cat] # we can just use the one from the first condition as they are all the same
            cur_lab = ['p1-p0', 'p2-p0']
            t_values=[]
            clus=[]
            clus_ps=[]
            for cmb in range(len(cur_X)):
                t_thresh = st.t.ppf(1 - analysis['mvpa']['cbp_clalpha'] / 2, df=X['z'].shape[0]-1) # cluster p = .005 (two-tailed)
                print(f"t_thresh = {round(t_thresh,3)}")
                T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                    X               = cur_X[cmb],   # data to be clustered
                    out_type        = "mask",                    # boolean arrays where true indicates part of a cluster
                    n_permutations  = analysis['mvpa']['cbp_nperm'],                 # number of permutations to compute
                    threshold       = t_thresh,                # so-called “cluster forming threshold” or threshold-free cluster enhancement (TFCE)
                    exclude         = cur_sel,
                    tail            = 0,   # two-tailed
                    seed            = 0
                )
                # save the data in list
                t_values.append(T_obs)
                clus.append(clusters)
                clus_ps.append(cluster_p_values)
                # Print significant clusters
                significant_clusters = np.where(cluster_p_values < 0.05)[0]
                nonsig_clusters = np.where(cluster_p_values >= 0.05)[0]
                print("-----")
                print(f"\n{c_cat}: {cur_lab[cmb]}")
                if significant_clusters.size == 0:
                    print("No significant clusters found")
                    for cl_idx in nonsig_clusters:
                        print(f"NON-significant cluster (p-value = {cluster_p_values[cl_idx]:.4f}):")
                        print(f"NON-significant time points: {subj['SVM_time'][clusters[cl_idx][0].start]} - {subj['SVM_time'][clusters[cl_idx][0].stop]} ")
                else:
                    for cl_idx in significant_clusters:
                        # Display the significant channels, frequencies, and time points
                        print(f"Significant cluster (p-value = {cluster_p_values[cl_idx]:.4f}):")
                        print(f"Significant time points: {subj['SVM_time'][clusters[cl_idx][0].start]} - {subj['SVM_time'][clusters[cl_idx][0].stop]} ")
                    for cl_idx in nonsig_clusters:
                        print(f"NON-significant cluster (p-value = {cluster_p_values[cl_idx]:.4f}):")
                        print(f"NON-significant time points: {subj['SVM_time'][clusters[cl_idx][0].start]} - {subj['SVM_time'][clusters[cl_idx][0].stop]} ")
                print("-----\n")
            # save the data to file
            # save the results
            clus_results[c_cat] = {
                                't_values'      : t_values,
                                'clusters'      : clus,
                                'cluster_ps'    : clus_ps,
                                'time_window'   : subj['SVM_time'],
                                }
            
            # Save X
            npz_fname = os.path.join(
                paths["outp_MEG"],
                analysis['type'],
                f"CBP_X_SVM_{c_cat}.npz"
            )
            np.savez_compressed(
                npz_fname,
                **{f"X_{k}": X[k] for k in X if k != "z"},
                X_z=X["z"]
            )
            meta = {
                "c_cat": c_cat,
                "conditions": c,
                "time_window": subj['SVM_time'],
                "subject_dim": list(svm_scores.coords["subject"].values),
            }

            json_fname = os.path.join(
                paths["outp_MEG"],
                analysis['type'],
                f"CBP_X_SVM_{c_cat}_meta.json"
            )

            with open(json_fname, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=to_json_safe)

            # PLOTTING
            # Classification
            times = subj['SVM_time']
            yloctext = .58 # y coordinate of the text in the figure
            sztext = 10 # size of text on figure
            ax={};ax[0]=[]
            h=[]
            h2=[]
            plt.rc('font',size = 15)
            plt.rcParams['axes.grid'] = False
            seaborn.set_palette(analysis['pltcol'])
            figsz = (15, 10)
            fig = plt.figure(dpi=300)
            fig.set_figheight(figsz[0])
            fig.set_figwidth(figsz[1])
            spec = gridspec.GridSpec(ncols=1, nrows=2,
                        width_ratios=[1], wspace=0,
                        hspace=.6, height_ratios=[1, 1])
            ax1 = fig.add_subplot(spec[0])
            ax[0] = fig.add_subplot(spec[1])
            for prime in range(len(c)):
                ax1.plot(
                    times,
                    np.mean(X[c[prime]],0),
                    label=task['pr_lab'][prime],
                )
            ax1.set_title("Decoding object category", pad=figsz[1]*1.5)
            ax1.set_ylabel("AUC")
            ax1.set_xlim(-.1,.5)
            ax1.set_ylim(.48,.6)
            ax1.axvline(x=-.7,color='0', linestyle='--'); ax1.axvline(x=-.45,color='0', linestyle='--'); ax1.axvline(x=-.35,color='0', linestyle='--'); ax1.axvline(x=-.1,color='0', linestyle='--'); ax1.axvline(x=0,color='0', linestyle='--'); ax1.axvline(x=.5,color='0', linestyle='--')
            ax1.axvline(x=.15,color='grey', linestyle='dotted'); ax1.axvline(x=.21,color='grey', linestyle='dotted')
            ax1.axvspan(0.15, 0.21, color='grey', alpha=0.3, hatch='xx', fill=False)
            ax1.axhline(.5, color='k', linestyle='--')
            ax1.legend(bbox_to_anchor =(0.5,-.3), loc='lower center', ncol=3)
            # Difference
            diff_lab = ["1-back prime - no prime", "2-back prime - no prime"]
            yloctext = 8.5 # y coordinate of the text in the figure
            for c2 in reversed(range(len(cur_X))): # reverse to make purple on the bottom
                # adjust the range if two clusters are very close to make them touch (less than 10 points)
                if sum(clus_ps[c2] < 0.05) == 2:
                    idx = [i for i, x in enumerate(clus_ps[c2] < 0.05) if x]
                    if clus[c2][idx[1]][0].start - clus[c2][idx[0]][0].stop < 10:
                        clus[c2][idx[0]] = (slice(clus[c2][idx[0]][0].start, clus[c2][idx[1]][0].stop, None),)
                        clus_ps[c2][idx[0]] = 1
                for i_c, cl in enumerate(clus[c2]):
                    if type(cl)==slice:
                        pass
                    else:
                        cl = cl[0]
                    if clus_ps[c2][i_c] < 0.05:
                        h  = ax[0].axvspan(times[cl.start], times[cl.stop - 1], color=analysis['pltcol'][1:][c2], alpha=0.3)
                    else:
                        pass
            for c2 in range(len(cur_X)): # reverse to make purple on the bottom
                ax[0].plot(times, t_values[c2], color=analysis['pltcol'][1:][c2], label=diff_lab[c2])
            ax[0].legend(bbox_to_anchor =(0.5,-.3), loc='lower center', ncol=2)
            ax[0].set_title('Condition Differences (t-values)', pad=figsz[1]*1.5)
            ax[0].set_xlabel("time (s)")
            ax[0].set_ylabel("t-values")
            ax[0].set_ylim(-8,8)
            ax[0].set_xlim(-.1,.5)
            ax[0].axvline(x=-.7,color='0', linestyle='--'); ax[0].axvline(x=-.45,color='0', linestyle='--'); ax[0].axvline(x=-.35,color='0', linestyle='--'); ax[0].axvline(x=-.1,color='0', linestyle='--'); ax[0].axvline(x=0,color='0', linestyle='--'); ax[0].axvline(x=.5,color='0', linestyle='--')
            ax[0].axvline(x=.15,color=analysis['pltcol'][1], linestyle='dotted'); ax[0].axvline(x=.21,color=analysis['pltcol'][1], linestyle='dotted')
            ax[0].axvspan(0.15, 0.21, color=analysis['pltcol'][1], alpha=0.3, hatch='xx', fill=False)
            ax[0].axhline(0, color='k', linestyle='--')
            fig
            # add label numbering
            fig.text(x = 0, y = .95, s = "A", fontsize = 40)
            fig.text(x = 0, y = .465, s = "B", fontsize = 40)
            #fig.set_size_inches((10, 20), forward=True)
            fig.savefig(os.path.join(paths["outp_MEG"],analysis['type'], cur_name), bbox_inches='tight', dpi=300)

        # get the time points of signficant p-values
        print('\n*** CBP: Time points of significant p-values ***\n')
        for c in combs:
            if c == ['p0', 'p1', 'p2']:
                c_cat = 'all'
            else:
                c_cat = c[0][-3:]
            print('\n'+c_cat+'\n')
            for i in range(len(cur_lab)):
                # find the significant cluster(s)
                c_sig = list(compress(clus_results[c_cat]['clusters'][i], clus_results[c_cat]['cluster_ps'][i]<.05))
                for cl in c_sig:
                    cl[0]
                    print('Significant p-value timepoints (',cur_lab[i],'):', 
                        subj['SVM_time'][cl[0].start],
                        subj['SVM_time'][cl[0].stop])
        # save SVM results
        fname = os.path.join(paths["outp_MEG"],analysis['type'], 'CBP_results_SVM.json')
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(clus_results, f, indent=2, default=to_json_safe)

        ## ERF-guided F-test and follow-up t-tests
        print('\n*** ERF-guided F-test and follow-up t-tests ***\n')
        t_win_rs = [0.15, 0.21] # repetition suppression
        t_win_re = [0.30, 0.36] # repetition enhancement
        cur_data = {}
        for c in combs:
            Fdata = {}
            Fdata_re = {}
            Fdata['ppn']=ppns_incl
            Fdata_re['ppn']=ppns_incl
            if c == ['p0', 'p1', 'p2']:
                c_cat = 'all'
            else:
                c_cat = c[0][-3:]
            print('\n'+c_cat+'\n')
            for p in c:
                da_p = (
                    svm_scores
                    .sel(prime=p)
                    .mean(dim="cat") # average over categories
                    .transpose("subject", "time")
                )
                mask_rs = (subj["SVM_time"] >= t_win_rs[0]) & (subj["SVM_time"] <= t_win_rs[1])
                mask_re = (subj["SVM_time"] >= t_win_re[0]) & (subj["SVM_time"] <= t_win_re[1])
                Fdata[p] = da_p.values[:, mask_rs].mean(axis=1)
                Fdata_re[p] = da_p.values[:, mask_re].mean(axis=1)

            # run the statistical test
            Fdata_w     = pd.DataFrame.from_dict(Fdata)
            Fdata_re_w  = pd.DataFrame.from_dict(Fdata_re)
            Fdata_l     = pd.melt(Fdata_w,id_vars="ppn", value_vars=list(Fdata_w.columns)[1::], var_name="cond", value_name="AUC", ignore_index=False) #long

            # Run repeated measures ANOVA with pingouin for effect size
            pg_results = pg.rm_anova(data=Fdata_l, dv="AUC", within='cond', subject='ppn', correction=True, detailed=True)
            cols = [
                "Source",
                "DF",
                "F",
                "p-GG-corr",
                "eps",
                "ng2", # ng2 = generalized eta squared
            ]

            print(pg_results[cols])
            row = pg_results.loc[pg_results["Source"] == "cond"].iloc[0]
            print(
                f"Condition effect: "
                f"F({row['DF']:.2f}) = {row['F']:.2f}, "
                f"p = {row['p-GG-corr']:.5f}, "
                f"ε_GG = {row['eps']:.3f}, "
                f"η²_G = {row['ng2']:.3f}"
            )
            
            p_val = row["p-GG-corr"]
            # Check if ANOVA is significant
            if p_val < 0.05:
                print('There was a significant difference between conditions')          
                print('\nFollow-up t-tests showed:')

                comparisons = [(c[1], c[0]), (c[2], c[0])]

                for cond1, cond2 in comparisons:
                    data1 = Fdata[cond1]
                    data2 = Fdata[cond2]

                    # Paired t-test
                    t_statistic, p_value = st.ttest_rel(data1, data2)
                    df = len(data1) - 1

                    # Effect size and CI using pingouin
                    stats = pg.ttest(data1, data2, paired=True, alternative='two-sided')
                    cohens_d = stats['cohen-d'].iloc[0]
                    ci_low, ci_high = stats['CI95%'].iloc[0]

                    # Means and SDs
                    mean1 = np.mean(data1)
                    mean2 = np.mean(data2)
                    sd1 = np.std(data1, ddof=1)
                    sd2 = np.std(data2, ddof=1)

                    if p_value < 0.05:
                        print(f'There was a significant difference between {cond1} and {cond2} (t({df}) = {t_statistic:.2f}, p = {p_value:.3f})')
                    else:
                        print(f'There was no significant difference between {cond1} and {cond2} (t({df}) = {t_statistic:.2f}, p = {p_value:.3f})')

                    print(f"Mean {cond1} = {mean1:.2f} (SD = {sd1:.2f})")
                    print(f"Mean {cond2} = {mean2:.2f} (SD = {sd2:.2f})")
                    print(f"Cohen's d = {cohens_d:.2f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]\n")

            else:
                print('There was no significant difference between conditions')

            ## Behav & SVM corr
            print('\n*** ERF-guided Memory & SVM Correlations ***\n')
            # transform the memory values to a dataframe
            subj_behav_mem_df = pd.DataFrame(subj_behav_mem)
            # merge the dataframes
            Fdata_w.columns = [Fdata_w.columns[0]] + ['svm_rs_' + col for col in Fdata_w.columns[1:]]
            Fdata_re_w.columns = [Fdata_re_w.columns[0]] + ['svm_re_' + col for col in Fdata_re_w.columns[1:]]
            mem_svm_df = pd.merge(pd.merge(Fdata_w, Fdata_re_w, on='ppn'), subj_behav_mem_df, on='ppn')
            # Calculate the differences of interst
            mem_svm_diff_df = pd.DataFrame()
            mem_svm_diff_df['svm_rs_p1-p0'] = mem_svm_df['svm_rs_p1'] - mem_svm_df['svm_rs_p0']
            mem_svm_diff_df['svm_rs_p2-p0'] = mem_svm_df['svm_rs_p2'] - mem_svm_df['svm_rs_p0']
            mem_svm_diff_df['svm_re_p1-p0'] = mem_svm_df['svm_re_p1'] - mem_svm_df['svm_re_p0']
            mem_svm_diff_df['svm_re_p2-p0'] = mem_svm_df['svm_re_p2'] - mem_svm_df['svm_re_p0']
            mem_svm_diff_df['mem_p1-p0'] = mem_svm_df['Memory_dprime_p0b'] - mem_svm_df['Memory_dprime_p1b']
            mem_svm_diff_df['mem_p2-p0'] = mem_svm_df['Memory_dprime_p0b'] - mem_svm_df['Memory_dprime_p2b']
            # Calculate correlations
            print('\n* Repetition Supression time window *\n')
            corr_p1, pval_p1 = spearmanr(mem_svm_diff_df['svm_rs_p1-p0'], mem_svm_diff_df['mem_p1-p0'])
            corr_p2, pval_p2 = spearmanr(mem_svm_diff_df['svm_rs_p2-p0'], mem_svm_diff_df['mem_p2-p0'])
            print(f"svm_rs_p1-p0 vs mem_p1-p0: Correlation = {corr_p1:.4f}, p-value = {pval_p1:.4g}")
            print(f"svm_rs_p2-p0 vs mem_p2-p0: Correlation = {corr_p2:.4f}, p-value = {pval_p2:.4g}")
            print('\n* Repetition Enhancement time window *\n')
            corr_p1, pval_p1 = spearmanr(mem_svm_diff_df['svm_re_p1-p0'], mem_svm_diff_df['mem_p1-p0'])
            corr_p2, pval_p2 = spearmanr(mem_svm_diff_df['svm_re_p2-p0'], mem_svm_diff_df['mem_p2-p0'])
            print(f"svm_re_p1-p0 vs mem_p1-p0: Correlation = {corr_p1:.4f}, p-value = {pval_p1:.4g}")
            print(f"svm_re_p2-p0 vs mem_p2-p0: Correlation = {corr_p2:.4f}, p-value = {pval_p2:.4g}")
        plt.close('all')

  
#%% BEHAVIORAL ANALYSES
if analysis['types']['behav']:
    print('\nBEHAVIORAL ANALYSES\n')
    # subjects with more than 50% of excluded trials in both ERFTFR and SVM analysis
    subj={}
    bad_subs = ['105','133']
    # empty dictionary to hold data
    mem_perf    = {"ppn":[],"dprime_p0":[],"dprime_p1":[],"dprime_p2":[]}
    mem_guess   = {"ppn":[],"Nguess_p0":[],"Nguess_p1":[],"Nguess_p2":[]}
    RTs         = {"ppn":[],"RT_p0":[],"RT_p1":[],"RT_p2":[]}
    descs       = {"ppn":[],"sex":[],"age":[]}
    for subj['nr'] in sorted(os.listdir(paths['inp_MEG'])):
        subj = {
            'nr'        :subj['nr'],
            'bad'       :False
            }
        # SAVED VARIABLES LOCATIONS
        var_loc = {
                    'sub' : os.path.join(paths["outp_MEG"],analysis['type'],subj['nr']+'_sub.json'), # where the subject data is saved
                    'beh' : os.path.join(paths["outp_BEH"],'behav_data.csv'), # where the subject data is saved
                    }
        # load in the data per participant
        with open(var_loc["sub"], "r", encoding="utf-8") as fp:
                subj = json.load(fp)
        if len([True for key, val in subj.items() if 'Behav' in key])==0:
        #if subj:
            #%% EXTRACT SUBJECT-LEVEL DATA
            subj['Behav'] = {}
            try: # remove the old way of storing the memory performance, for conciseness
                subj.pop('Memory_dprime_all')
                subj.pop('Memory_dprime_p0b')
                subj.pop('Memory_dprime_p1b')
                subj.pop('Memory_dprime_p2b')
            except:
                pass
            # import the retrieval files
            ret_files = os.listdir(paths['inp_BEH'])
            ret = pd.read_csv(os.path.join(paths['inp_BEH'], fnmatch.filter(ret_files, 'Ret_' + subj['nr'] + '*.csv')[0]))
            # GET DESCRIPTIVES
            if ret.gender[0][0].lower() == 'f':
                subj['Behav']['sex'] = 0
            elif ret.gender[0][0].lower() == 'o':
                subj['Behav']['sex'] = 1
            elif ret.gender[0][0].lower() == 'm':
                subj['Behav']['sex'] = 2
            else:
                raise Exception("gender not predefined, please check")
            subj['Behav']['age'] = ret.age[0]
            # COMPUTE MEMORY PERFORMANCE
            # all
            hitrate = sum(ret.memcat==128) / (sum(ret.memcat==128) + sum(ret.memcat==32)) # hits devided by hits + misses
            farate  = sum(ret.memcat==16) / (sum(ret.memcat==16) + sum(ret.memcat==64)) # false alarms devided by false alarms + correct rejections
            subj['Behav']['Memory_dprime_all']  = norm.ppf(hitrate) - norm.ppf(farate)
            # not primed (prime0b)
            hitrate = sum(np.logical_and(ret.memcat==128, ret.cond=='prime0b')) / (sum(np.logical_and(ret.memcat==128, ret.cond=='prime0b')) + sum(np.logical_and(ret.memcat==32, ret.cond=='prime0b'))) # prime0b hits devided by prime0b hits + prime0b misses
            farate  = sum(ret.memcat==16) / (sum(ret.memcat==16) + sum(ret.memcat==64)) # false alarms devided by false alarms + correct rejections
            subj['Behav']['Memory_dprime_p0b']  = norm.ppf(hitrate) - norm.ppf(farate)
            # primed 1 screen back (prime1b)
            hitrate = sum(np.logical_and(ret.memcat==128, ret.cond=='prime1b')) / (sum(np.logical_and(ret.memcat==128, ret.cond=='prime1b')) + sum(np.logical_and(ret.memcat==32, ret.cond=='prime1b'))) # prime1b hits devided by prime1b hits + prime1b misses
            farate  = sum(ret.memcat==16) / (sum(ret.memcat==16) + sum(ret.memcat==64)) # false alarms devided by false alarms + correct rejections
            subj['Behav']['Memory_dprime_p1b']  = norm.ppf(hitrate) - norm.ppf(farate)
            # primed 2 screens back (prime1b)
            hitrate = sum(np.logical_and(ret.memcat==128, ret.cond=='prime2b')) / (sum(np.logical_and(ret.memcat==128, ret.cond=='prime2b')) + sum(np.logical_and(ret.memcat==32, ret.cond=='prime2b'))) # prime2b hits devided by prime2b hits + prime2b misses
            farate  = sum(ret.memcat==16) / (sum(ret.memcat==16) + sum(ret.memcat==64)) # false alarms devided by false alarms + correct rejections
            subj['Behav']['Memory_dprime_p2b']  = norm.ppf(hitrate) - norm.ppf(farate)
            # REACTION TIMES
            subj['Behav']['RT_all'] = np.nanmean(ret.RT)
            subj['Behav']['RT_p0b'] = np.nanmean(ret.RT[ret.cond=='prime0b'])
            subj['Behav']['RT_p1b'] = np.nanmean(ret.RT[ret.cond=='prime1b'])
            subj['Behav']['RT_p2b'] = np.nanmean(ret.RT[ret.cond=='prime2b'])
            # GUESSES
            subj['Behav']['Memory_guess_all'] = sum(ret.memcat==8) + sum(ret.memcat==4) # guess (old) (8), guess (new) (4)
            subj['Behav']['Memory_guess_p0b'] = sum(np.logical_and(ret.memcat==8, ret.cond=='prime0b')) + sum(np.logical_and(ret.memcat==4, ret.cond=='prime0b')) # guess (old) (8), guess (new) (4)
            subj['Behav']['Memory_guess_p1b'] = sum(np.logical_and(ret.memcat==8, ret.cond=='prime1b')) + sum(np.logical_and(ret.memcat==4, ret.cond=='prime1b')) # guess (old) (8), guess (new) (4)
            subj['Behav']['Memory_guess_p2b'] = sum(np.logical_and(ret.memcat==8, ret.cond=='prime2b')) + sum(np.logical_and(ret.memcat==4, ret.cond=='prime2b')) # guess (old) (8), guess (new) (4)
            # save the subject data
            with open(var_loc["sub"], "w", encoding="utf-8") as fp:
                json.dump(subj, fp, indent=2, default=to_json_safe)
        # only add the subject to the grand average if it is not marked as bad
        descs["ppn"].append(int(subj["nr"]))
        descs['sex'].append(subj['Behav']['sex'])
        descs['age'].append(subj['Behav']['age'])
        if not subj['nr'] in bad_subs:
            #print(subj['nr'])
            # paste the data in the dictionary
            mem_perf["ppn"].append(int(subj["nr"]))
            mem_perf["dprime_p0"].append(subj['Behav']["Memory_dprime_p0b"])
            mem_perf["dprime_p1"].append(subj['Behav']["Memory_dprime_p1b"])
            mem_perf["dprime_p2"].append(subj['Behav']["Memory_dprime_p2b"])
            # paste the data in the dictionary
            RTs["ppn"].append(int(subj["nr"]))
            RTs["RT_p0"].append(subj['Behav']["RT_p0b"])
            RTs["RT_p1"].append(subj['Behav']["RT_p1b"])
            RTs["RT_p2"].append(subj['Behav']["RT_p2b"])
            # paste the data in the dictionary
            mem_guess["ppn"].append(int(subj["nr"]))
            mem_guess["Nguess_p0"].append(subj['Behav']["Memory_guess_p0b"])
            mem_guess["Nguess_p1"].append(subj['Behav']["Memory_guess_p1b"])
            mem_guess["Nguess_p2"].append(subj['Behav']["Memory_guess_p2b"])
    #%% PERFORM GROUP-LEVEL ANALYSES
    # descriptives
    print('\n*** Descriptives ***')
    print('\nMean age = ' + "{:.2f}".format(np.mean(descs['age'])) + ' (SD = ' + "{:.2f}".format(np.std(descs['age']))+')')
    print('\nN female = '+str(sum(np.equal(descs['sex'],0)))+'\nN male = '+str(sum(np.equal(descs['sex'],2)))+'\n')
    # transform the data into a dataframe
    mem_perf_w  = pd.DataFrame.from_dict(mem_perf) # wide format
    mem_perf_l  = pd.melt(mem_perf_w,id_vars="ppn", value_vars=list(mem_perf_w.columns)[1::], var_name="cond", value_name="dprime", ignore_index=False) #long
    mem_perf_diff = {} # difference scores
    mem_perf_diff['ppn'] = mem_perf_w['ppn']
    mem_perf_diff['dprime_p1p0'] = mem_perf_w['dprime_p1'] - mem_perf_w['dprime_p0']
    mem_perf_diff['dprime_p2p0'] = mem_perf_w['dprime_p2'] - mem_perf_w['dprime_p0']
    mem_perf_diff_w = pd.DataFrame.from_dict(mem_perf_diff) # wide format
    mem_perf_diff_l = pd.melt(mem_perf_diff_w,id_vars="ppn", value_vars=list(mem_perf_diff_w.columns)[1::], var_name="cond", value_name="dprime", ignore_index=False) #long
    RTs_w       = pd.DataFrame.from_dict(RTs) # wide format
    RTs_l       = pd.melt(RTs_w,id_vars="ppn", value_vars=list(RTs_w.columns)[1::], var_name="cond", value_name="RT", ignore_index=False) #long
    mem_guess_w  = pd.DataFrame.from_dict(mem_guess) # wide format
    mem_guess_l  = pd.melt(mem_guess_w,id_vars="ppn", value_vars=list(mem_guess_w.columns)[1::], var_name="cond", value_name="Nguess", ignore_index=False) #long
    mem_guess_diff = {} # difference scores
    mem_guess_diff['ppn'] = mem_guess_w['ppn']
    mem_guess_diff['Nguess_p1p0'] = mem_guess_w['Nguess_p1'] - mem_guess_w['Nguess_p0']
    mem_guess_diff['Nguess_p2p0'] = mem_guess_w['Nguess_p2'] - mem_guess_w['Nguess_p0']
    mem_guess_diff_w = pd.DataFrame.from_dict(mem_guess_diff) # wide format
    mem_guess_diff_l = pd.melt(mem_guess_diff_w,id_vars="ppn", value_vars=list(mem_guess_diff_w.columns)[1::], var_name="cond", value_name="Nguess", ignore_index=False) #long
    
    # SAVE DATA
    # prepare the data for saving and plots
    behav_data = {}
    behav_data["mem_perf_w"] = mem_perf_w
    behav_data["mem_perf_l"] = mem_perf_l
    behav_data["mem_perf_diff_w"] = mem_perf_diff_w
    behav_data["mem_perf_diff_l"] = mem_perf_diff_l
    behav_data["RTs_w"] = RTs_w
    behav_data["RTs_l"] = RTs_l
    behav_data["mem_guess_w"] = mem_guess_w
    behav_data["mem_guess_l"] = mem_guess_l
    behav_data["mem_guess_diff_w"] = mem_guess_diff_w
    behav_data["mem_guess_diff_l"] = mem_guess_diff_l
    # further prepare the data for saving as we aren't using pickle anymore
    wide_dfs = {
        k: v for k, v in behav_data.items()
        if k.endswith("_w")
    }
    wide_merged = functools.reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on="ppn",
            how="left",
            validate="one_to_one"
        ),
        wide_dfs.values()
    )
    # save the data
    wide_merged.to_csv(
        var_loc['beh'],
        index=False,
    )

    # Repeated measures ANOVA with pingouin
    pg_results = pg.rm_anova(data=behav_data["mem_perf_l"], dv='dprime', within='cond', subject='ppn', correction=True, detailed=True)

    cols = [
        "Source",
        "DF",
        "F",
        "p-GG-corr",
        "eps",
        "ng2", # ng2 = generalized eta squared
    ]
    print(pg_results[cols])
    row = pg_results.loc[pg_results["Source"] == "cond"].iloc[0]
    print(
        f"Condition effect: "
        f"F({row['DF']:.2f}) = {row['F']:.2f}, "
        f"p = {row['p-GG-corr']:.5f}, "
        f"ε_GG = {row['eps']:.3f}, "
        f"η²_G = {row['ng2']:.3f}"
    )

    p_val = row["p-GG-corr"]
    # Post hoc tests if ANOVA is significant
    if p_val < 0.05:
        print('\nPost hoc comparisons with effect sizes and confidence intervals:\n')

        comparisons = [("dprime_p1", "dprime_p0", "Prime one back vs No Prime"),
                    ("dprime_p2", "dprime_p0", "Prime two back vs No Prime")]

        p_ttest_plot = {}
        for cond1, cond2, label in comparisons:
            data1 = behav_data["mem_perf_w"][cond1]
            data2 = behav_data["mem_perf_w"][cond2]

            # Paired t-test
            t_stat, p_ttest = st.ttest_rel(data1, data2)
            df = len(data1) - 1

            # Effect size (Cohen's d) and confidence interval
            stats = pg.ttest(data1, data2, paired=True, alternative='two-sided')

            d = stats['cohen-d'].iloc[0]
            ci_low = stats['CI95%'].iloc[0][0]
            ci_high = stats['CI95%'].iloc[0][1]

            print(f'\n* {label} *')
            print(f't({df}) = {t_stat:.2f}, p = {p_ttest:.3f}')
            print(f'Cohen\'s d = {d:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]')
            
            # Means and standard deviations
            mean1 = np.mean(data1)
            mean2 = np.mean(data2)
            sd1 = np.std(data1, ddof=1)
            sd2 = np.std(data2, ddof=1)

            print(f'Mean {cond1} = {mean1:.2f} (SD = {sd1:.2f})')
            print(f'Mean {cond2} = {mean2:.2f} (SD = {sd2:.2f})')

            p_ttest_plot[cond1] = p_ttest

        # difference
        comparisons = [("dprime_p1p0", "dprime_p2p0", "Prime one back - No Prime vs Prime two back - No Prime")]
        for cond1, cond2, label in comparisons:
            data1 = behav_data["mem_perf_diff_w"][cond1]
            data2 = behav_data["mem_perf_diff_w"][cond2]
            # Paired t-test
            t_stat, p_ttest = st.ttest_rel(data1, data2)
            df = len(data1) - 1

            # Effect size (Cohen's d) and confidence interval
            stats = pg.ttest(data1, data2, paired=True, alternative='two-sided')

            d = stats['cohen-d'].iloc[0]
            ci_low = stats['CI95%'].iloc[0][0]
            ci_high = stats['CI95%'].iloc[0][1]

            print(f'\n* {label} *')
            print(f't({df}) = {t_stat:.2f}, p = {p_ttest:.3f}')
            print(f'Cohen\'s d = {d:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]')
            
            # Means and standard deviations
            mean1 = np.mean(data1)
            mean2 = np.mean(data2)
            sd1 = np.std(data1, ddof=1)
            sd2 = np.std(data2, ddof=1)

            print(f'Mean {cond1} = {mean1:.2f} (SD = {sd1:.2f})')
            print(f'Mean {cond2} = {mean2:.2f} (SD = {sd2:.2f})')

    # PLOTTING
    # plot the memory performance
    try:
        seaborn.set_theme(style="whitegrid",font_scale=3)
        ax = seaborn.violinplot(data=mem_perf_l, x="cond", y="dprime", palette=analysis['pltcol'], saturation=1, inner=None, legend=False, linewidth=5, linecolor="black")
        ax = seaborn.swarmplot(data=mem_perf_l, x="cond", y="dprime", palette=analysis['pltcol'], size=10, edgecolor="black", linewidth=2)
        plt.setp(ax.collections, alpha=.75)
        cur_means = mem_perf_l.groupby('cond')['dprime'].mean()
        plt.axhline(xmin=1/6-.05, xmax=1/6+.05, y=cur_means["dprime_p0"], color="black", linestyle='-', linewidth=5)
        plt.axhline(xmin=3/6-.05, xmax=3/6+.05, y=cur_means["dprime_p1"], color="black", linestyle='-', linewidth=5)
        plt.axhline(xmin=5/6-.05, xmax=5/6+.05, y=cur_means["dprime_p2"], color="black", linestyle='-', linewidth=5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([c[-2:] for c in list(np.unique(mem_perf_l['cond']))])
        ax.set_xlabel(xlabel = "Conditions",labelpad=30)
        ax.set_ylabel(ylabel = "Memory performance (d')",labelpad=30)
        ax.axhline(y=0,color='black', linestyle="--",label='chance')
        ax.legend(loc='upper right')
        fig = ax.get_figure() 
        try:
            fig.set_size_inches((20, 20), forward=True)
        except:
            print('Could not alter the size of the figure')
        fig.savefig(os.path.join(paths["outp_MEG"], 'Mem_dprimes.png'), bbox_inches='tight', dpi=300)
        plt.close('all')
    except:
        print('Could not make and/or save the figure')
    # plot the difference in memory performance
    try:
        seaborn.set_theme(style="whitegrid", font_scale=3, rc={"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False})
        ax = seaborn.violinplot(data=mem_perf_diff_l, x="cond", y="dprime", palette=analysis['pltcol'][1:], saturation=1, inner=None, legend=False, linewidth=5, linecolor="black")
        ax = seaborn.swarmplot(data=mem_perf_diff_l, x="cond", y="dprime", palette=analysis['pltcol'][1:], size=10, edgecolor="black", linewidth=2)
        ax.axhline(y=0,color='black', linestyle="--", linewidth=2)
        plt.setp(ax.collections, alpha=.75)
        cur_means = mem_perf_diff_l.groupby('cond')['dprime'].mean()
        plt.axhline(xmin=1/4-.05, xmax=1/4+.05, y=cur_means["dprime_p1p0"], color="black", linestyle='-', linewidth=5)
        plt.axhline(xmin=3/4-.05, xmax=3/4+.05, y=cur_means["dprime_p2p0"], color="black", linestyle='-', linewidth=5)
        plt.ylim(-.85, .85) 
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["1-back prime - no prime", "2-back prime - no prime"])
        ax.set_xlabel(xlabel = None)
        #ax.set_xlabel(xlabel = "Conditions",labelpad=30)
        ax.set_ylabel(ylabel = "Difference in memory performance (d')",labelpad=30)
        # Add statistical annotations
        y_adj = 0
        p = p_ttest_plot['dprime_p1']
        if p <= 1.00e-04:
            sig_text = "****"
        elif 1.00e-04 < p <= 1.00e-03:
            sig_text = "***"
        elif 1.00e-03 < p <= 1.00e-02:
            sig_text = "**"
        elif 1.00e-02 < p <= 5.00e-02:
            sig_text = "*"
        elif 5.00e-02 < p <= 1.00e+00:
            sig_text = "n.s."
            y_adj = .03
        ax.text(
            x=0, y=.81+y_adj, s=sig_text, ha='center', fontsize="x-large",
            color='black'
        )
        # Add statistical annotations
        y_adj = 0
        p = p_ttest_plot['dprime_p2']
        if p <= 1.00e-04:
            sig_text = "****"
        elif 1.00e-04 < p <= 1.00e-03:
            sig_text = "***"
        elif 1.00e-03 < p <= 1.00e-02:
            sig_text = "**"
        elif 1.00e-02 < p <= 5.00e-02:
            sig_text = "*"
        elif 5.00e-02 < p <= 1.00e+00:
            sig_text = "n.s."
            y_adj = .03
        ax.text(
            x=1, y=.81+y_adj, s=sig_text, ha='center', fontsize="small",
            color='black'
        )
        annotator = Annotator(ax, [("dprime_p1p0", "dprime_p2p0")], data=mem_perf_diff_l, x="cond", y="dprime")
        annotator.configure(test="t-test_paired", text_format="star", loc="outside", fontsize="x-large")
        annotator.apply_and_annotate()
        fig = ax.get_figure() 
        try:
            fig.set_size_inches((20, 20), forward=True)
        except:
            print('Could not alter the size of the figure')
        fig.savefig(os.path.join(paths["outp_MEG"], 'Mem_dprimes_diff.png'), bbox_inches='tight', dpi=300)
        plt.close('all')
    except:
        print('Could not make and/or save the figure')
    # plot the RTs
    try:
        seaborn.set_theme(style="whitegrid",font_scale=3)
        ax = seaborn.violinplot(data=RTs_l, x="cond", y="RT", palette=analysis['pltcol'], saturation=1, inner=None, legend=False, linewidth=5, linecolor="black")
        ax = seaborn.swarmplot(data=RTs_l, x="cond", y="RT", palette=analysis['pltcol'], size=10, edgecolor="black", linewidth=2)
        plt.setp(ax.collections, alpha=.75)
        cur_means = RTs_l.groupby('cond')['RT'].mean()
        plt.axhline(xmin=1/6-.05, xmax=1/6+.05, y=cur_means["RT_p0"], color="black", linestyle='-', linewidth=5)
        plt.axhline(xmin=3/6-.05, xmax=3/6+.05, y=cur_means["RT_p1"], color="black", linestyle='-', linewidth=5)
        plt.axhline(xmin=5/6-.05, xmax=5/6+.05, y=cur_means["RT_p2"], color="black", linestyle='-', linewidth=5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([c[-2:] for c in list(np.unique(RTs_l['cond']))])
        ax.set_xlabel(xlabel = "Conditions",labelpad=30)
        ax.set_ylabel(ylabel = "Mean RT (s)",labelpad=30)
        ax.axhline(y=0,color='black', linestyle="--",label='chance')
        ax.legend(loc='upper right')
        fig = ax.get_figure() 
        try:
            fig.set_size_inches((20, 20), forward=True)
        except:
            print('Could not alter the size of the figure')
        fig.savefig(os.path.join(paths["outp_MEG"], 'RTs.png'), bbox_inches='tight', dpi=300)
        plt.close('all')
    except:
        print('Could not make and/or save the figure')


