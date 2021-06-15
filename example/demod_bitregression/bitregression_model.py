import os, sys
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# to run this file from within example/demod_bitregression folder
os.chdir(os.getcwd())
sys.path.append(os.curdir)

sig_len = 40960

def get_model(window_len=1024, n_ch=2):
    inputs = tf.keras.Input(shape=(window_len, n_ch))

    h = layers.Flatten()(inputs)
    h = layers.Dense(window_len, activation='relu')(h)

    for _ in range(5):
        h2 = layers.Dense(window_len, activation='relu')(h)
        h2 = layers.Dense(window_len, activation='relu')(h2)
        h = layers.Add()([h, h2])

    outputs = layers.Dense(window_len//16*2, activation='sigmoid')(h)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bit_regression")

    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002),
        metrics=["binary_accuracy"],
    )

    return model


window_len = 1024
n_ch = 2

interference_sig_type = "EMISignal1"
emi1_model = get_model(window_len, n_ch)
emi1_model.load_weights(os.path.join('example','demod_bitregression','models',f'demod_regression_{interference_sig_type}_{window_len}')).expect_partial()


interference_sig_type = "CommSignal2"
comm2_model = get_model(window_len, n_ch)
comm2_model.load_weights(os.path.join('example','demod_bitregression','models',f'demod_regression_{interference_sig_type}_{window_len}')).expect_partial()


interference_sig_type = "CommSignal3"
comm3_model = get_model(window_len, n_ch)
comm3_model.load_weights(os.path.join('example','demod_bitregression','models',f'demod_regression_{interference_sig_type}_{window_len}')).expect_partial()


def demod_bits(sig_mixture, model=None, interference_sig_type=None, window_len=1024):
    if model is None:
        assert interference_sig_type in ["EMISignal1", "CommSignal2", "CommSignal3"], f"Models only trained for EMISignal1, CommSignal2 or CommSignal3; invalid interference_sig_type: {interference_sig_type}"
        assert window_len == 1024, f"Models only available window_len=1024, invalid window_len: {window_len}"
        
        if interference_sig_type == "EMISignal1":
            model = emi1_model
        elif interference_sig_type == "CommSignal2":
            model = comm2_model
        elif interference_sig_type == "CommSignal3":
            model = comm3_model

    x_in = sig_mixture.reshape(-1, window_len)
    x_in_comp = np.stack((x_in.real, x_in.imag), axis=-1)
    
    bit_probs = model.predict(x_in_comp)
    bit_est = np.array(bit_probs > 0.5).flatten()
    return bit_est