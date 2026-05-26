
# setup
import os
import h5py
import pandas as pd
import yaml
from pathlib import Path

config_file_path = Path.cwd() / "configs" / "data.yaml"
with open(config_file_path, "r") as file:
    config = yaml.safe_load(file)

def h5_info(name, obj):
    """Prints the path, object type, shape, and data type for HDF5 contents."""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: /{name:<25} | Shape: {str(obj.shape):<15} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group:   /{name:<25}")

# read data
waveform_h5_path = os.path.join(config['data_dir'], 'mimic_iv_ecg_waveforms.h5')
label_h5_path = os.path.join(config['data_dir'], 'mimic_iv_ecg_icd.h5')
metadata_csv_path = os.path.join(config['data_dir'], 'metadata.csv')

f_wave = h5py.File(waveform_h5_path, 'r', rdcc_nbytes=1024*1024*4)
f_label = h5py.File(label_h5_path, 'r')
df = pd.read_csv(metadata_csv_path)

# info
f_wave.visititems(h5_info)
f_label.visititems(h5_info)
print(df.info())
