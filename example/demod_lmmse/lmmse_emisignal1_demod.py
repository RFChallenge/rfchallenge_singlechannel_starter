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

import random
random.seed(0)
np.random.seed(0)

val_or_test = 'val'
interference_sig_type = 'EMISignal1'

output_folder = os.path.join('example', 'demod_lmmse', 'output')

def main():
    dataset_type = f'demod_{val_or_test}'
    all_ber, all_default_ber, all_sdr, all_sinr = [], [], [], []
    
    all_test_idx = np.arange(1100)
    for idx in all_test_idx:
        sig_mixture,meta = rfcutils.load_dataset_sample(idx, dataset_type, interference_sig_type)
        sig1,meta1,sig2,meta2 = rfcutils.load_dataset_sample_components(idx, dataset_type, interference_sig_type)
        
        sinr = get_sinr(sig1, sig2)
        all_sinr.append(sinr)
        ber_ref = rfcutils.demod_check_ber(rfcutils.matched_filter_demod(sig_mixture), idx, dataset_type, interference_sig_type)
        all_default_ber.append(ber_ref)
        
        sig1_mmse, sig2_mmse = lmmse_separate(sig_mixture, soi_type='QPSK', interference_sig_type=interference_sig_type)
        bit_est = rfcutils.matched_filter_demod(sig1_mmse)
        ber = rfcutils.demod_check_ber(bit_est, idx, dataset_type, interference_sig_type)
        all_ber.append(ber)
        
        sdr = get_sinr(sig1, sig1-sig1_mmse)
        all_sdr.append(sdr)
        
        print(f"#{idx} -- SINR {sinr:.3f}dB: 1:{ber} Default:{ber_ref}, SDR:{sdr}")
        
        if len(all_sinr)%100 == 0:
            pickle.dump((all_ber, all_default_ber, all_sdr, all_sinr), open(os.path.join(output_folder, f'lmmse_{interference_sig_type}_{val_or_test}_demod.pickle'),'wb'))
    pickle.dump((all_ber, all_default_ber, all_sdr, all_sinr), open(os.path.join(output_folder, f'lmmse_{interference_sig_type}_{val_or_test}_demod.pickle'),'wb'))
    
if __name__ == "__main__":
    main()