import os
import numpy as np
import datetime as dt

import sigmf
from sigmf import SigMFFile, sigmffile

contact_email = 'aia-rf@mit.edu'


# adapted from "Use Case" in gnuradio/SigMF repository (https://github.com/gnuradio/SigMF)

def read_sigmf_file(filename=f'sample', folderpath=None):

    # Load a dataset
    full_filename = filename
    if folderpath is not None:
        full_filename = os.path.join(folderpath, filename)

    meta = sigmffile.fromfile(full_filename)

    # Get some metadata and all annotations
    sample_rate = meta.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    sample_count = meta.sample_count
    signal_duration = sample_count / sample_rate

    data = meta.read_samples(0, meta.sample_count)
    return data, meta


def write_sigmf_file(data, filename=f'sample', folderpath=None, fs=1, fc=0, description=''):
    assert data.dtype == 'complex64'

    full_filename = filename
    if folderpath is not None:
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        full_filename = os.path.join(folderpath, filename)

    # write those samples to file in cf32_le
    data.tofile(f"{full_filename}.sigmf-data")

    # create the metadata
    meta = SigMFFile(
        data_file=f"{full_filename}.sigmf-data", # extension is optional
        global_info = {
            SigMFFile.DATATYPE_KEY: 'cf32_le',
            SigMFFile.SAMPLE_RATE_KEY: fs,
            SigMFFile.AUTHOR_KEY: contact_email,
            SigMFFile.DESCRIPTION_KEY: description,
            SigMFFile.VERSION_KEY: sigmf.__version__,
        }
    )

    # create a capture key at time index 0
    meta.add_capture(0, metadata={
        SigMFFile.FREQUENCY_KEY: fc,
        SigMFFile.DATETIME_KEY: dt.datetime.utcnow().isoformat()+'Z',
    })

    # check for mistakes & write to disk
    assert meta.validate()
    meta.tofile(f"{full_filename}.sigmf-meta") # extension is optional

    return data, meta
