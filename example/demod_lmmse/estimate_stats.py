import os, sys

# to run this file from within example/demod_lmmse folder
os.chdir(os.getcwd())
print(os.path.abspath(os.curdir))
sys.path.append(os.curdir)

import numpy as np
import pickle
import rfcutils

import random
random.seed(0)
np.random.seed(0)

stats_folder = os.path.join('example', 'demod_lmmse', 'stats')
block_len = 4000
window_len=40960

def load_train_frames(interference_type, num_train_instances):
    interference_sig_dataset = []
    for ii in range(num_train_instances):
        data,meta = rfcutils.load_dataset_sample(ii, 'train_frame', interference_type)
        interference_sig_dataset.append(data)
    interference_sig_dataset = np.array(interference_sig_dataset)
    return interference_sig_dataset

def generate_samples(interference_sig_dataset, block_len=block_len, window_len=window_len):
    block_dataset = []
    for ii in range(10000):
        idx = np.random.randint(interference_sig_dataset.shape[0])
        start_idx = np.random.randint(interference_sig_dataset.shape[1]-window_len)
        train_data = interference_sig_dataset[idx, start_idx:start_idx+window_len]
        train_data = train_data/np.mean(np.abs(train_data)**2)
        train_data = train_data[:(window_len//block_len)*block_len]
        block_data = train_data.reshape(-1,block_len)
        block_dataset.append(block_data)
    block_dataset = np.array(block_dataset)
    block_dataset = block_dataset.reshape(-1,block_len)
    return block_dataset


def generate_aligned_samples(sig_dataset, block_len=block_len, window_len=window_len):
    template_start = None
    
    block_dataset = []
    for ii in range(10000):
        idx = np.random.randint(sig_dataset.shape[0])
        start_idx = np.random.randint(sig_dataset.shape[1]-window_len)
        train_data = sig_dataset[idx, start_idx:start_idx+window_len]
        train_data = train_data/np.mean(np.abs(train_data)**2)

        # alignment
        if ii == 0:
            template_start = train_data[:4000]
        else:
            pk_idx = np.argmax(np.abs(np.correlate(train_data,template_start,mode='full')))
            pk_idx -= len(template_start)
            train_data = np.roll(train_data, -pk_idx)
        
        train_data = train_data[:(window_len//block_len)*block_len]
        block_data = train_data.reshape(-1,block_len)
        block_dataset.append(block_data)
    block_dataset = np.array(block_dataset)
    block_dataset = block_dataset.reshape(-1,block_len)
    return block_dataset, template_start


interference_type = 'EMISignal1'
interference_sig_dataset = load_train_frames(interference_type, 530)
block_dataset = generate_samples(interference_sig_dataset)
mu_emi1 = np.mean(block_dataset, axis=0)
cov1_emi1 = 1/block_dataset.shape[0]*np.matmul(np.transpose(np.conj(block_dataset-mu_emi1)), block_dataset-mu_emi1)
cov2_emi1 = 1/block_dataset.shape[0]*np.matmul(np.transpose((block_dataset-mu_emi1)), block_dataset-mu_emi1)

pickle.dump((mu_emi1,cov1_emi1,cov2_emi1),open(os.path.join(stats_folder,f'{interference_type}_stats.pickle'),'wb'))


interference_type = 'CommSignal2'
interference_sig_dataset = load_train_frames(interference_type, 100)
block_dataset = generate_samples(interference_sig_dataset)
mu_comm2 = np.mean(block_dataset, axis=0)
cov1_comm2 = 1/block_dataset.shape[0]*np.matmul(np.transpose(np.conj(block_dataset-mu_comm2)), block_dataset-mu_comm2)
cov2_comm2 = 1/block_dataset.shape[0]*np.matmul(np.transpose((block_dataset-mu_comm2)), block_dataset-mu_comm2)

pickle.dump((mu_comm2,cov1_comm2,cov2_comm2),open(os.path.join(stats_folder,f'{interference_type}_stats.pickle'),'wb'))


interference_type = 'CommSignal3'
interference_sig_dataset = load_train_frames(interference_type, 139)
block_dataset = generate_samples(interference_sig_dataset)
mu_comm3 = np.mean(block_dataset, axis=0)
cov1_comm3 = 1/block_dataset.shape[0]*np.matmul(np.transpose(np.conj(block_dataset-mu_comm3)), block_dataset-mu_comm3)
cov2_comm3 = 1/block_dataset.shape[0]*np.matmul(np.transpose((block_dataset-mu_comm3)), block_dataset-mu_comm3)

pickle.dump((mu_comm3,cov1_comm3,cov2_comm3),open(os.path.join(stats_folder,f'{interference_type}_stats.pickle'),'wb'))



sig_type = 'QPSK'
qpsk_block_dataset = []
for ii in range(10000):
    qpsk_sig, _, _, _ = rfcutils.generate_qpsk_signal()
    qpsk_sig = qpsk_sig[:(len(qpsk_sig)//block_len)*block_len]
    block_data = qpsk_sig.reshape(-1,block_len)
    qpsk_block_dataset.append(block_data)
qpsk_block_dataset = np.array(qpsk_block_dataset)
qpsk_block_dataset = qpsk_block_dataset.reshape(-1,block_len)

mu_qpsk = np.mean(qpsk_block_dataset, axis=0)
cov1_qpsk = 1/qpsk_block_dataset.shape[0]*np.matmul(np.transpose(np.conj(qpsk_block_dataset-mu_qpsk)), qpsk_block_dataset-mu_qpsk)
cov2_qpsk = 1/qpsk_block_dataset.shape[0]*np.matmul(np.transpose((qpsk_block_dataset-mu_qpsk)), qpsk_block_dataset-mu_qpsk)

pickle.dump((mu_qpsk,cov1_qpsk,cov2_qpsk),open(os.path.join(stats_folder,f'{sig_type}_demod_stats.pickle'),'wb'))


##########
sig_type = 'CommSignal2'
sig_sig_dataset = load_train_frames(sig_type, 100)
block_dataset, template_start = generate_aligned_samples(sig_sig_dataset)
mu_comm2 = np.mean(block_dataset, axis=0)
cov1_comm2 = 1/block_dataset.shape[0]*np.matmul(np.transpose(np.conj(block_dataset-mu_comm2)), block_dataset-mu_comm2)
cov2_comm2 = 1/block_dataset.shape[0]*np.matmul(np.transpose((block_dataset-mu_comm2)), block_dataset-mu_comm2)

pickle.dump((mu_comm2,cov1_comm2,cov2_comm2,template_start),open(os.path.join(stats_folder,f'{sig_type}_aligned_stats.pickle'),'wb'))