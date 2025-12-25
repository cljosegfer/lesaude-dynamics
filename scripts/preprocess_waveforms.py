import pandas as pd
import numpy as np
import wfdb
import h5py
import os
from tqdm import tqdm

from hparams import DATA_ROOT

# ================= CONFIGURATION =================
CSV_FILE = 'record_list.csv'
H5_PATH = 'mimic_iv_ecg_waveforms.h5'

# Baseline Constants (Strodthoff et al.)
TARGET_LEN = 5000  # 10s @ 500Hz
CHANNELS = 12
EPSILON = 1e-6     # For numerical stability
# =================================================

def robust_z_score(signal):
    """
    Matches the AI4HealthUOL baseline:
    - Z-score per lead.
    - Handles flat leads (std=0) by setting them to 0.
    - Replaces NaNs with 0.
    """
    # 1. Handle NaNs first (Crucial for MIMIC)
    signal = np.nan_to_num(signal)
    
    # 2. Calculate Stats per lead (axis 0 is time, 1 is leads)
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    
    # 3. Normalize
    # Where std is effectively 0, we just return the signal (which is mean-centered 0)
    # avoiding division by zero artifacts
    normalized = np.zeros_like(signal)
    
    for lead in range(signal.shape[1]):
        if std[lead] > EPSILON:
            normalized[:, lead] = (signal[:, lead] - mean[lead]) / std[lead]
        else:
            # If line is flat, keep it as 0s (mean subtracted)
            normalized[:, lead] = 0.0
            
    return normalized

def build_monolith_baseline():
    print(f"Reading {CSV_FILE}...")
    df = pd.read_csv(os.path.join(DATA_ROOT, CSV_FILE))
    
    # SORTING: Crucial for your Latent Dynamics "Next Step" logic
    # We sort by Subject -> Time so pairs are neighbors in the HDF5
    df['ecg_time'] = pd.to_datetime(df['ecg_time'])
    df = df.sort_values(by=['subject_id', 'ecg_time'])
    
    records = df.to_dict('records')
    total_records = len(records)
    
    print(f"Building Monolith for {total_records} records...")
    
    with h5py.File(os.path.join(DATA_ROOT, H5_PATH), 'w') as f:
        # 1. Waveform Dataset (Float32 for precision, or Float16 to save ~50% space)
        # Strodthoff baseline uses float32. 
        dset_waves = f.create_dataset('waveforms', 
                                      shape=(total_records, TARGET_LEN, CHANNELS), 
                                      dtype='float32', 
                                      chunks=(1, TARGET_LEN, CHANNELS)) 
        
        # 2. Subject ID (Integer for fast filtering)
        dset_subj = f.create_dataset('subject_id', shape=(total_records,), dtype='i4')
        dset_study = f.create_dataset('study_id', shape=(total_records,), dtype='i4')
        
        # 3. Timestamps (Stored as UNIX timestamp for math: t_2 - t_1)
        dset_time = f.create_dataset('ecg_time', shape=(total_records,), dtype='f8')

        success_idx = 0
        valid_indices = [] # To filter the CSV later

        for i, row in tqdm(enumerate(records), total=total_records):
            # Resolve Path (Handle partial paths from CSV)
            # CSV path usually: files/p1000/p10000032/s40689238/40689238
            rel_path = row['path']
            full_path_no_ext = os.path.join(DATA_ROOT, rel_path)
            
            try:
                # Read Signal
                record = wfdb.rdrecord(full_path_no_ext)
                signal = record.p_signal
                
                # Check Dimensions
                length, n_leads = signal.shape
                if n_leads != CHANNELS:
                    print(i, rel_path, row, " - Unexpected number of leads:", n_leads)
                    continue # Skip 3-lead or malformed files
                
                # Resize to 5000 (10s)
                if length > TARGET_LEN:
                    # Baseline takes FIRST 10 seconds, not center crop
                    signal = signal[:TARGET_LEN, :]
                    print(i, rel_path, row, " - Warning: Truncated signal from", length, "to", TARGET_LEN)
                elif length < TARGET_LEN:
                    # Zero pad end
                    pad_len = TARGET_LEN - length
                    signal = np.pad(signal, ((0, pad_len), (0, 0)), 'constant')
                    print(i, rel_path, row, " - Warning: Padded signal from", length, "to", TARGET_LEN)
                
                # Apply Baseline Normalization
                signal = robust_z_score(signal)
                
                # Save
                dset_waves[success_idx] = signal
                dset_subj[success_idx] = int(row['subject_id'])
                dset_study[success_idx] = int(row['study_id'])
                dset_time[success_idx] = row['ecg_time'].timestamp()
                
                valid_indices.append(i)
                success_idx += 1
                
            except Exception as e:
                # Missing files (if you filtered the zip) are skipped here
                print(i, rel_path, row, " - Error reading file:", e)
                continue

        # Resize HDF5 to actual count
        # dset_waves.resize((success_idx, TARGET_LEN, CHANNELS))
        # dset_subj.resize((success_idx,))
        # dset_study.resize((success_idx,))
        # dset_time.resize((success_idx,))
        
        print(f"\nFinal Count: {success_idx} / {total_records}")
    
    # # Save the "Aligned" CSV
    # print("Saving Aligned CSV...")
    # df_clean = df.iloc[valid_indices].reset_index(drop=True)
    # df_clean['h5_index'] = df_clean.index
    # df_clean.to_csv('mimic_ecg_monolith_metadata.csv', index=False)
    print("Done.")

if __name__ == "__main__":
    build_monolith_baseline()