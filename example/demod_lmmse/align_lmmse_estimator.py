import os
import numpy as np
import pickle
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s1, s2: 10*np.log10(get_pow(s1)/get_pow(s2))

stats_folder = os.path.join('example', 'demod_lmmse', 'stats')

soi_type = 'QPSK'
mu_qpsk,cov1_qpsk,cov2_qpsk = pickle.load(open(os.path.join(stats_folder,f'{soi_type}_demod_stats.pickle'),'rb'))
mu_qpsk = mu_qpsk.reshape(-1,1)


interference_sig_type = 'CommSignal2'
mu_comm2,cov1_comm2,cov2_comm2,template_start = pickle.load(open(os.path.join(stats_folder,f'{interference_sig_type}_aligned_stats.pickle'),'rb'))
mu_comm2 = mu_comm2.reshape(-1,1)



def align_lmmse_separate(sig_mixture, soi_type='QPSK', interference_sig_type = 'EMISignal1'):
    assert soi_type == 'QPSK', f"This align_lmmse_separate function is created for QPSK + CommSignal2. Ensure that soi_type is 'QPSK' -- soi_type provided: {soi_type}"
    if soi_type == 'QPSK':
        Css = cov1_qpsk
        mu_s = mu_qpsk
        
    assert interference_sig_type == 'CommSignal2', f"This align_lmmse_separate function is created for QPSK + CommSignal2. Ensure that interference_sig_type is 'CommSignal2' -- interference_sig_type provided: {interference_sig_type}"
    if interference_sig_type == 'CommSignal2':
        Cbb = cov1_comm2
        mu_b = mu_comm2
        
    
    pk_idx = np.argmax(np.abs(np.correlate(sig_mixture,template_start,mode='full')))
    pk_idx -= len(template_start)
    pk_idx = pk_idx % Cbb.shape[0]
    Cbb_roll = np.roll(np.roll(Cbb, pk_idx, axis=0), pk_idx, axis=1)
    mu_b_roll = np.roll(mu_b, pk_idx, axis=0)
    
    Cbb = Cbb_roll
    mu_b = mu_b_roll
    
    est_sinr = 1/(get_pow(sig_mixture)-1)
    scaled_Cbb = Cbb * 1/est_sinr
    Csy = np.vstack((Css, scaled_Cbb))
    Cyy = Css + scaled_Cbb
    U,S,Vh = np.linalg.svd(Cyy,hermitian=True)
#     Cyy_inv = np.matmul(U, np.matmul(np.diag(1.0/(S + 1e-4)), U.conj().T))
#     Cyy_inv = np.linalg.pinv(Cyy,hermitian=True)
    sthr_idx = np.where(S>1e-4)[0][-1]
    Cyy_inv = np.matmul(U[:,:sthr_idx], np.matmul(np.diag(1.0/(S[:sthr_idx])), U[:,:sthr_idx].conj().T))
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
    
    return sig1_mmse, sig2_mmse


