import os
import warnings
import numpy as np
import pickle

from .sigmf_helper_fn import write_sigmf_file, read_sigmf_file

def load_dataset_sample(idx, dataset_type, sig_type):
    foldername = os.path.join('dataset',dataset_type,sig_type)
    filename = f'{sig_type}_{dataset_type}_{idx:04d}'
    # Special handling for "Separation" validation and test set; only using Comm2 vs [sig_type] for this iteration
    if 'sep_' in dataset_type:
        filename = f'CommSignal2_vs_{sig_type}_{dataset_type}_{idx:04d}'
    data, meta = read_sigmf_file(filename=filename, folderpath=foldername)
    return data, meta

def load_dataset_sample_components(idx, dataset_type, sig_type):
    assert 'train' in dataset_type or 'val' in dataset_type or 'test' in dataset_type, f'Invalid dataset type requested for obtaining components: {dataset_type}'
    
    soi_name = 'Comm2' if 'sep_' in dataset_type else 'QPSK'
    foldername1 = os.path.join('dataset',dataset_type,'Components', sig_type, soi_name)
    filename1 = f'{sig_type}_{dataset_type}_{idx:04d}'
    # Special handling for "Separation" validation and test set; only using Comm2 vs [sig_type] for this iteration
    if 'sep_' in dataset_type:
        filename1 = f'CommSignal2_vs_{sig_type}_{dataset_type}_{idx:04d}'
    data1, meta1 = read_sigmf_file(filename=filename1, folderpath=foldername1)
    
    foldername2 = os.path.join('dataset',dataset_type,'Components', sig_type, 'Interference')
    filename2 = f'{sig_type}_{dataset_type}_{idx:04d}'
    # Special handling for "Separation" validation and test set; only using Comm2 vs [sig_type] for this iteration
    if 'sep_' in dataset_type:
        filename2 = f'CommSignal2_vs_{sig_type}_{dataset_type}_{idx:04d}'
    data2, meta2 = read_sigmf_file(filename=filename1, folderpath=foldername2)
    
    return data1, meta1, data2, meta2

def load_dataset_sample_demod_groundtruth(idx, dataset_type, sig_type):
    assert 'train' in dataset_type or 'val' in dataset_type or 'test' in dataset_type, f'Invalid dataset type requested for obtaining components: {dataset_type}'
    assert 'demod_' in dataset_type, f'Invalid dataset type requested for obtaining components: {dataset_type}'
    
    foldername = os.path.join('dataset',dataset_type,'Components',sig_type,'QPSK')
    filename = f'{sig_type}_{dataset_type}_{idx:04d}'
    data, meta = read_sigmf_file(filename=filename, folderpath=foldername)

    msg_folder = os.path.join('dataset',dataset_type,'QPSK_Bits',sig_type)
    msg_filename = f'{sig_type}_{dataset_type}_QPSK_bits_{idx:04d}'
    msg_bits, ground_truth_info = pickle.load(open(os.path.join(msg_folder,f'{msg_filename}.pkl'),'rb'))
    return data, meta, msg_bits, ground_truth_info


def demod_check_ber(bit_est, idx, dataset_type, sig_type):
    assert 'demod_' in dataset_type, f'Invalid dataset type requested for obtaining components: {dataset_type}'
    
    msg_folder = os.path.join('dataset',dataset_type,'QPSK_Bits',sig_type)
    msg_filename = f'{sig_type}_{dataset_type}_QPSK_bits_{idx:04d}'
    bit_true, _ = pickle.load(open(os.path.join(msg_folder,f'{msg_filename}.pkl'),'rb'))
    if len(bit_est) != len(bit_true):
        warnings.warn(f'Mismatch in estimated bit message length ({len(bit_est)}) and true bit message length ({len(bit_true)})')
        msg_len = min(len(bit_true), len(bit_est))
        bit_true = bit_true[:msg_len]
        bit_est = bit_est[:msg_len]
    ber = np.sum(np.abs(bit_est-bit_true))/len(bit_true)
    return ber
