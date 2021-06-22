import os, sys
import numpy as np
import pickle
from tqdm import tqdm

from bitregression_model import get_model
import tensorflow as tf

# to run this file from within example/demod_bitregression folder
os.chdir(os.getcwd())
print(os.path.abspath(os.curdir))
sys.path.append(os.curdir)

import rfcutils
get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_sinr2 = lambda s, i: (np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))

import random

dataset_type = f'demod_train'
n_train_frame_types = {'EMISignal1':530, 'CommSignal2':100, 'CommSignal3':139}

sig_len = 40960
window_len = 1024

for interference_sig_type in ['EMISignal1', 'CommSignal2', 'CommSignal3']:
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    n_train_frame = n_train_frame_types[interference_sig_type]
    
    x_in, sig1_out, sig2_out, bit_out = [], [] , [], []

    # for idx in tqdm(range(1100)):
    #     sig_mixture,meta = rfcutils.load_dataset_sample(idx, dataset_type, interference_sig_type)
    #     sig1,meta1,sig2,meta2 = rfcutils.load_dataset_sample_components(idx, dataset_type, interference_sig_type)

    #     bit_info = rfcutils.matched_filter_demod(sig1)

    #     for l in range(40960//window_len):
    #         x_in.append(sig_mixture[l*window_len:(l+1)*window_len])
    #         sig1_out.append(sig1[l*window_len:(l+1)*window_len])
    #         sig2_out.append(sig2[l*window_len:(l+1)*window_len])
    #         bit_out.append(bit_info[l*window_len//16*2:(l+1)*window_len//16*2])

    for idx in tqdm(range(4000)):
        for target_sinr in np.arange(-12,4.5,1.5):
            chosen_idx = np.random.randint(n_train_frame)
            data,meta = rfcutils.load_dataset_sample(chosen_idx, 'train_frame', interference_sig_type)
            start_idx = np.random.randint(len(data)-sig_len)
            sig2 = data[start_idx:start_idx+sig_len]

            sig1, _, _, bit_info = rfcutils.generate_qpsk_signal()

            coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))

            sig_mixture = sig1 + sig2 * coeff

            for l in range(40960//window_len):
                x_in.append(sig_mixture[l*window_len:(l+1)*window_len])
                sig1_out.append(sig1[l*window_len:(l+1)*window_len])
                sig2_out.append(sig2[l*window_len:(l+1)*window_len])
                bit_out.append(bit_info[l*window_len//16*2:(l+1)*window_len//16*2])
            
    x_in = np.array(x_in)
    # sig1_out = np.array(sig1_out)
    # sig2_out = np.array(sig2_out)
    bit_out = np.array(bit_out)
    x_in_comp = np.stack((x_in.real, x_in.imag), axis=-1)
    
    _, window_len, n_ch = x_in_comp.shape

    model = get_model(window_len, n_ch)
    model.fit(x_in_comp, bit_out, batch_size=32, epochs=200, verbose=1, shuffle=True)
    model.save_weights(os.path.join('models',f'demod_regression_{interference_sig_type}_{window_len}'))