import numpy as np
import random

from .sigmf_helper_fn import write_sigmf_file, read_sigmf_file
from .dataset_helper_fn import load_dataset_sample
from .qpsk_helper_fn import generate_qpsk_signal

window_len = 40960
get_rand_start_idx = lambda sig_len: np.random.randint(sig_len-window_len)


num_train_frame = {"EMISignal1": 530, "CommSignal2": 100, "CommSignal3": 139}
num_trainval_frame = {"EMISignal1": 580, "CommSignal2": 150, "CommSignal3": 189}

def create_sep_mixture(sig_type, target_sinr_db, seed=None, dataset_type="train"):
    np.random.seed(seed)
    random.seed(seed)
    
    if dataset_type == "all" or dataset_type == "val":
        comm2_chosen_idx = np.random.randint(num_trainval_frame["CommSignal2"])
    elif dataset_type == "train":
        comm2_chosen_idx = np.random.randint(num_train_frame["CommSignal2"])
    
    comm2_data, comm2_meta = load_dataset_sample(comm2_chosen_idx, 'train_frame', 'CommSignal2')
    
    
    if dataset_type == "all":
        chosen_idx = np.random.randint(num_trainval_frame[sig_type])
    elif dataset_type == "train":
        chosen_idx = np.random.randint(num_train_frame[sig_type])
    elif dataset_type == "val":
        if comm2_chosen_idx < num_train_frame["CommSignal2"]:
            # mixture should be from an unseen pool:
            chosen_idx = np.random.randint(num_train_frame[sig_type], num_trainval_frame[sig_type])
    
    data, meta = load_dataset_sample(chosen_idx, 'train_frame', sig_type)
            
            
    comm2_start_idx = get_rand_start_idx(len(comm2_data))            
    sig1 = comm2_data[comm2_start_idx:comm2_start_idx+window_len]
    coeff1 = np.sqrt(1/(np.mean(np.abs(sig1)**2)))
    sig1 *= coeff1
            
    start_idx = get_rand_start_idx(len(data))
    sig2 = data[start_idx:start_idx+window_len]

    coeff2 = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr_db/10)))) 
    sig2 *= coeff2
    
    sig_mixture = sig1 + sig2
    return sig_mixture, sig1, sig2
    


oversample_factor = 16
n_sym = window_len//oversample_factor

def create_demod_mixture(sig_type, target_sinr_db, seed=None, dataset_type="train"):
    np.random.seed(seed)
    random.seed(seed)
    
    qpsk_sig, _, qpsk_sym, msg_bits = generate_qpsk_signal(n_sym)
    
    if dataset_type == "all":
        chosen_idx = np.random.randint(num_trainval_frame[sig_type])
    elif dataset_type == "train":
        chosen_idx = np.random.randint(num_train_frame[sig_type])
    elif dataset_type == "val":
        chosen_idx = np.random.randint(num_train_frame[sig_type], num_trainval_frame[sig_type])
    
    data, meta = load_dataset_sample(chosen_idx, 'train_frame', sig_type)
    
    
    sig1 = qpsk_sig
            
    start_idx = get_rand_start_idx(len(data))
    sig2 = data[start_idx:start_idx+window_len]

    coeff2 = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr_db/10)))) 
    sig2 *= coeff2
    
    sig_mixture = sig1 + sig2
    return sig_mixture, sig1, sig2, msg_bits
    