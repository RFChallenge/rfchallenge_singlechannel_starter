import os, sys
import numpy as np
import pickle
from tqdm import tqdm
import time

# to run this file from within example/demod_bitregression folder
os.chdir(os.getcwd())
print(os.path.abspath(os.curdir))
sys.path.append(os.curdir)

import rfcutils
get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))

import random
random.seed(0)
np.random.seed(0)

val_or_test = 'val'
interference_sig_type = 'EMISignal1'

def main(): 
    dataset_type = f'demod_{val_or_test}'
    
    all_default_ber, all_sinr = [], []
    all_test_idx = np.arange(1100)
    for idx in all_test_idx:
        sig_mixture,meta = rfcutils.load_dataset_sample(idx, dataset_type, interference_sig_type)
        sig1,meta1,sig2,meta2 = rfcutils.load_dataset_sample_components(idx, dataset_type, interference_sig_type)
        
        sinr = get_sinr(sig1, sig2)
        all_sinr.append(sinr)
        ber_ref = rfcutils.demod_check_ber(rfcutils.matched_filter_demod(sig_mixture), idx, dataset_type, interference_sig_type)
        all_default_ber.append(ber_ref)
        
        print(f"#{idx} -- SINR {sinr:.3f}dB: Default:{ber_ref}")
        
        if len(all_sinr)%100 == 0:
            pickle.dump((all_default_ber, all_sinr), open(os.path.join('example','demod_matched_filtering','output',f'matchedfiltering_{interference_sig_type}_{val_or_test}_demod.pickle'),'wb'))
    pickle.dump((all_default_ber, all_sinr), open(os.path.join('example','demod_matched_filtering','output',f'matchedfiltering_{interference_sig_type}_{val_or_test}_demod.pickle'),'wb'))
    
if __name__ == "__main__":
    main()
