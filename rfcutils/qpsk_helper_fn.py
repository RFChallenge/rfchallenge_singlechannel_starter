import os
import numpy as np
import pickle

from .sigmf_helper_fn import write_sigmf_file, read_sigmf_file

from scipy.signal import convolve
from commpy.modulation import PSKModem, QAMModem
from commpy.filters import rrcosfilter, rcosfilter


# Parameters for QPSK
mod_num = 4
mod = PSKModem(mod_num)

rolloff = 0.5
Fs = 25e6
oversample_factor = 16
Ts = oversample_factor/Fs
tVec, sPSF = rrcosfilter(oversample_factor*8, rolloff, Ts, Fs)
tVec, sPSF = tVec[1:], sPSF[1:]
sPSF = sPSF.astype(np.complex64)

seg_len = int(2**15 + 2**13)
n_sym = seg_len//oversample_factor

def generate_qpsk_signal(n_sym=n_sym, mod=mod, oversample_factor=oversample_factor, sPSF=sPSF, Fc=0, Fs=Fs):
    sB = np.random.randint(2, size=n_sym*mod.num_bits_symbol)  # Random bit stream
    sQ = mod.modulate(sB)  # Modulated baud points
    sQ_padded = np.zeros(len(sQ)*oversample_factor, dtype=np.complex64)
    start_idx = oversample_factor//2
    sQ_padded[start_idx::oversample_factor] = sQ

    sig = convolve(sQ_padded, sPSF, 'same') # Waveform with PSF
    sig *= np.exp(2*np.pi*1j*np.arange(len(sig))*Fc/Fs, dtype=np.complex64)
    return sig, sQ_padded, sQ, sB

def get_psf():
    return sPSF

def matched_filter(sig, sPSF, Fc=0, Fs=Fs):
    sig_filt = sig*np.exp(-2*np.pi*1j*np.arange(len(sig))*Fc/Fs, dtype=np.complex64)
    sig_filt = convolve(sig_filt, sPSF/np.sum(sPSF), 'same')
    return sig_filt

def matched_filter_demod(sig, sPSF=sPSF, Fc=0, Fs=Fs):
    sig_filt = matched_filter(sig, sPSF, Fc=0, Fs=Fs)
    sig_filt = sig_filt[oversample_factor//2::oversample_factor]
    bit_est = mod.demodulate(sig_filt, demod_type='hard')
#     noise_var = 0.1
#     bit_llr = mod.demodulate(sig_filt, demod_type='soft', noise_var=noise_var)
#     bit_prob = 1/(1+np.exp(bit_llr))
#     bit_prob = np.clip(bit_prob, 1e-12, 1-1e-12)
    return bit_est



def modulate_qpsk_signal(info_bits, mod=mod, oversample_factor=oversample_factor, sPSF=sPSF, Fc=0, Fs=Fs):
    sB = info_bits
    sQ = mod.modulate(sB)  # Modulated baud points
    sQ_padded = np.zeros(len(sQ)*oversample_factor, dtype=np.complex64)
    start_idx = oversample_factor//2
    sQ_padded[start_idx::oversample_factor] = sQ

    sig = convolve(sQ_padded, sPSF, 'same') # Waveform with PSF
    sig *= np.exp(2*np.pi*1j*np.arange(len(sig))*Fc/Fs, dtype=np.complex64)
    return sig