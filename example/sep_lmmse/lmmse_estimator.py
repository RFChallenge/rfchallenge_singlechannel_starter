import os
import numpy as np
import pickle
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s1, s2: 10*np.log10(get_pow(s1)/get_pow(s2))

stats_folder = os.path.join('example', 'sep_lmmse', 'stats')

interference_sig_type = 'CommSignal2'
mu_comm2a,cov1_comm2a,cov2_comm2a,template_start = pickle.load(open(os.path.join(stats_folder,f'{interference_sig_type}_aligned_stats.pickle'),'rb'))
mu_comm2a = mu_comm2a.reshape(-1,1)

interference_sig_type = 'CommSignal2'
mu_comm2,cov1_comm2,cov2_comm2 = pickle.load(open(os.path.join(stats_folder,f'{interference_sig_type}_stats.pickle'),'rb'))
mu_comm2 = mu_comm2.reshape(-1,1)


interference_sig_type = 'EMISignal1'
mu_emi1,cov1_emi1,cov2_emi1 = pickle.load(open(os.path.join(stats_folder,f'{interference_sig_type}_stats.pickle'),'rb'))
mu_emi1 = mu_emi1.reshape(-1,1)

interference_sig_type = 'CommSignal3'
mu_comm3,cov1_comm3,cov2_comm3 = pickle.load(open(os.path.join(stats_folder,f'{interference_sig_type}_stats.pickle'),'rb'))
mu_comm3 = mu_comm3.reshape(-1,1)


def lmmse_separate(sig_mixture, soi_type='CommSignal2', interference_sig_type = 'EMISignal1', align_soi=True):
    assert soi_type == 'CommSignal2', f"This lmmse_separate function is created for separation of CommSignal2 + Interference. Ensure that soi_type is 'QPSK' -- soi_type provided: {soi_type}"
    if soi_type == 'CommSignal2':
        if align_soi:
            Css = cov1_comm2a
            mu_s = mu_comm2a
        else:
            Css = cov1_comm2
            mu_s = mu_comm2
        
    assert interference_sig_type in ['EMISignal1', 'CommSignal3'], f"This lmmse_separate function is created for separation of CommSignal2 + Interference. Ensure that interference_sig_type is one of ['EMISignal1', 'CommSignal3'] -- interference_sig_type provided: {interference_sig_type}"
    if interference_sig_type == 'EMISignal1':
        Cbb = cov1_emi1
        mu_b = mu_emi1
    elif interference_sig_type == 'CommSignal3':
        Cbb = cov1_comm3
        mu_b = mu_comm3
    
    
    if align_soi:
        pk_idx = np.argmax(np.abs(np.correlate(sig_mixture,template_start,mode='full')))
        pk_idx -= len(template_start)
        sig_mixture = np.roll(sig_mixture, -pk_idx)
        
        
    est_sinr = 1/(get_pow(sig_mixture)-1)
    scaled_Cbb = Cbb * 1/est_sinr
    Csy = np.vstack((Css, scaled_Cbb))
    Cyy = Css + scaled_Cbb
    Cyy_inv = np.linalg.pinv(Cyy,hermitian=True)
    W = np.matmul(Csy,Cyy_inv)
    
    window_len = W.shape[1]
    original_len = len(sig_mixture)
    test_sig = np.zeros(int(window_len*np.ceil(original_len/window_len)), dtype=complex)
    test_sig[:original_len] = sig_mixture

    test_sig_input = test_sig.reshape(-1,window_len)
    est_components = np.matmul(W,test_sig_input.T - mu_s - mu_b)

    sig1_mmse = (est_components[:window_len,:] + mu_s).T.flatten()
    sig2_mmse = (est_components[window_len:,:] + mu_b).T.flatten()
    sig1_mmse = sig1_mmse[:original_len]
    sig2_mmse = sig2_mmse[:original_len]
    
    if align_soi:
        sig1_mmse = np.roll(sig1_mmse, pk_idx)
        sig2_mmse = np.roll(sig2_mmse, pk_idx)
    
    return sig1_mmse, sig2_mmse


