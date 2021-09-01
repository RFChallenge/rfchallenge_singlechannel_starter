import os, sys
import numpy as np
import pickle
from lmmse_estimator import lmmse_separate

# to run this file from within example/demod_lmmse folder
os.chdir(os.getcwd())
print(os.path.abspath(os.curdir))
sys.path.append(os.curdir)

import rfcutils
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s1, s2: 10*np.log10(get_pow(s1)/get_pow(s2))
get_mse = lambda s: np.mean(np.abs(s)**2)

import random
random.seed(0)
np.random.seed(0)

val_or_test = 'val'
interference_sig_type = 'EMISignal1'
align_soi = True
align_soi_str = 'aligned' if align_soi else 'not_aligned'

output_folder = os.path.join('example', 'sep_lmmse', 'output')

def main():
    dataset_type = f'sep_{val_or_test}'
    all_mse, all_default_mse, all_sdr, all_sinr = [], [], [], []
    
    all_test_idx = np.arange(1100)
    for idx in all_test_idx:
        sig_mixture,meta = rfcutils.load_dataset_sample(idx, dataset_type, interference_sig_type)
        sig1,meta1,sig2,meta2 = rfcutils.load_dataset_sample_components(idx, dataset_type, interference_sig_type)
        
        sinr = get_sinr(sig1, sig2)
        all_sinr.append(sinr)
        mse_ref = get_mse(sig_mixture-sig1)
        all_default_mse.append(mse_ref)
        
        sig1_mmse, sig2_mmse = lmmse_separate(sig_mixture, soi_type='CommSignal2', interference_sig_type=interference_sig_type, align_soi=align_soi)
        
        sdr = get_sinr(sig1, sig1-sig1_mmse)
        all_sdr.append(sdr)
        
        mse = get_mse(sig1_mmse-sig1)
        all_mse.append(mse)
        
        print(f"#{idx} -- SINR {sinr:.3f}dB: MSE:{mse} SDR:{sdr}, Mixture MSE:{get_pow(sig_mixture-sig1_mmse-sig2_mmse)}")
        
        if len(all_sinr)%100 == 0:
            pickle.dump((all_mse, all_default_mse, all_sdr, all_sinr), open(os.path.join(output_folder, f'lmmse_{interference_sig_type}_{val_or_test}_sep_{align_soi_str}.pickle'),'wb'))
    pickle.dump((all_mse, all_default_mse, all_sdr, all_sinr), open(os.path.join(output_folder, f'lmmse_{interference_sig_type}_{val_or_test}_sep_{align_soi_str}.pickle'),'wb'))
    
if __name__ == "__main__":
    main()