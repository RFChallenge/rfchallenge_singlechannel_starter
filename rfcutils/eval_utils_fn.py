import warnings
import numpy as np


def eval_ber(bit_est, bit_true):
    if len(bit_est) != len(bit_true):
        warnings.warn(f'Mismatch in estimated bit message length ({len(bit_est)}) and true bit message length ({len(bit_true)})')
        msg_len = min(len(bit_true), len(bit_est))
        bit_true = bit_true[:msg_len]
        bit_est = bit_est[:msg_len]
    ber = np.sum(np.abs(bit_est-bit_true))/len(bit_true)
    return ber

def eval_logloss(bit_prob, bit_true):
    if len(bit_prob) != len(bit_true):
        warnings.warn(f'Mismatch in estimated bit message length ({len(bit_prob)}) and true bit message length ({len(bit_true)})')
        msg_len = min(len(bit_true), len(bit_prob))
        bit_true = bit_true[:msg_len]
        bit_prob = bit_prob[:msg_len]
    logloss = -np.mean((bit_true==0)*np.log2(bit_prob) + (bit_true==1)*np.log2(1-bit_prob))
    return logloss
