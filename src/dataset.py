import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from hparams import DATA_ROOT

class DynamicsDataset(Dataset):
    def __init__(self, 
                 waveform_h5_path = os.path.join(DATA_ROOT, 'mimic_iv_ecg_waveforms.h5'),
                 label_h5_path = os.path.join(DATA_ROOT, 'mimic_iv_ecg_icd.h5'),
                 metadata_csv_path=os.path.join(DATA_ROOT, 'metadata.csv'),
                 split='train',
                 return_pairs=True,):
        """
        Args:
            waveform_h5_path (str): Path to mimic_iv_ecg_waveforms.h5
            label_h5_path (str): Path to mimic_iv_ecg_icd.h5
            metadata_csv_path (str): Path to metadata.csv
            split (str): 'train', 'val', or 'test'
            return_pairs (bool): If True, returns (x_t, x_t+1, a_t). If False, returns (x_t, y_t) for classification.
        """
        self.wave_path = waveform_h5_path
        self.label_path = label_h5_path
        self.return_pairs = return_pairs
        
        # File handles (initialized lazily to support num_workers > 0)
        self.f_wave = None
        self.f_label = None

        # folds
        df = pd.read_csv(metadata_csv_path)

        if split == 'train':
            target_folds = list(range(0, 18)) # 0 to 17
        elif split == 'val':
            target_folds = [18]
        elif split == 'test':
            target_folds = [19]
        elif split == 'all':
            target_folds = list(range(0, 20))
        else:
            raise ValueError(f"Unknown split: {split}")
        
        df_subset = df[df['fold'].isin(target_folds)].copy()
        if not return_pairs and split in ['val', 'test']:
            print(f"   > Applying Baseline Filter: Keeping only first ECG per stay (ecg_no_within_stay == 0)")
            original_len = len(df_subset)
            df_subset = df_subset[df_subset['ecg_no_within_stay'] == 0]
            print(f"   > Reduced {original_len} -> {len(df_subset)} records.")
        self.indices = df_subset['h5_index'].values
        
        print(f"Initializing Dataset from {waveform_h5_path}...")
        
        # --- Pre-calculate Valid Indices ---
        # We need to know which indices 'i' have a valid 'i+1' (Same Patient)
        # We perform a quick scan of the subject_ids.
        with h5py.File(self.wave_path, 'r') as fw, h5py.File(self.label_path, 'r') as fl:
            
            # Load Subject IDs into memory (Fast, int32 array is small)
            print("   Loading Subject IDs for integrity check...")
            subj_wave = fw['subject_id'][:]
            subj_label = fl['subject_id'][:]
            
            # 1. Integrity Check (Performed ONCE at startup)
            if not np.array_equal(subj_wave, subj_label):
                mismatch_idx = np.where(subj_wave != subj_label)[0][0]
                raise ValueError(f"CRITICAL DATA MISMATCH: Waveform and Label files are not aligned at index {mismatch_idx}. "
                                 f"Wave subj: {subj_wave[mismatch_idx]}, Label subj: {subj_label[mismatch_idx]}")
            print("   > Integrity Check Passed: Waveform and Label files are perfectly aligned.")

            # 2. Study ID Check (Optional, but good if you have the data)
            print("   Loading Study IDs for integrity check...")
            study_wave = fw['study_id'][:]
            study_label = fl['study_id'][:]
            if not np.array_equal(study_wave, study_label):
                raise ValueError("CRITICAL DATA MISMATCH: Study IDs do not match.")

            # 3. Calculate Indices
            if self.return_pairs:
                # Create a boolean mask of the whole dataset where mask[i] = True if i is in our split
                split_mask = np.zeros(len(subj_wave), dtype=bool)
                split_mask[self.indices] = True

                # Logic: Index 'i' is valid IF subject[i] == subject[i+1]
                same_patient_mask = (subj_wave[:-1] == subj_wave[1:])
                in_split = (split_mask[:-1] & split_mask[1:])
                valid_mask = same_patient_mask & in_split
                self.valid_indices = np.where(valid_mask)[0]
                
                # Balancing Logic
                print("   Scanning for action types...")
                # Load all labels (int8 is small enough for RAM)
                # Ensure the key matches your H5 file ('icd' or 'labels')
                all_labels = fl['icd'][:] 
                
                curr_labels = all_labels[self.valid_indices]
                next_labels = all_labels[self.valid_indices + 1]
                
                # Check for equality
                is_stable = np.all(curr_labels == next_labels, axis=1)
                
                self.stable_indices = self.valid_indices[is_stable]
                self.changed_indices = self.valid_indices[~is_stable]
                
                print(f"   > Total Pairs: {len(self.valid_indices)}")
                print(f"   > Stable Pairs: {len(self.stable_indices)}")
                print(f"   > Changed Pairs: {len(self.changed_indices)}")
                
            else:
                # self.valid_indices = np.arange(len(subj_wave))
                self.valid_indices = self.indices

    def _open_files(self):
        if self.f_wave is None:
            # rdcc_nbytes: Raw Data Chunk Cache size. 
            # Default is 1MB. Increase to 4MB or 8MB to smooth out reads.
            self.f_wave = h5py.File(self.wave_path, 'r', rdcc_nbytes=1024*1024*16)
            # self.f_wave = h5py.File(self.wave_path, 'r')
        if self.f_label is None:
            self.f_label = h5py.File(self.label_path, 'r')

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        self._open_files()
        
        # Map the dataset index (0..N) to the HDF5 index (which might skip patient boundaries)
        real_idx = self.valid_indices[idx]

        # 1. load pair
        pair_data = self.f_wave['waveforms'][real_idx : real_idx + 2]
        pair_labels = self.f_label['icd'][real_idx : real_idx + 2]

        x_t = torch.from_numpy(pair_data[0]).float()
        y_t = torch.from_numpy(pair_labels[0]).long()
        x_t = x_t.transpose(0, 1)
        if not self.return_pairs:
            # Classification Mode: Just return x, y
            return {'waveform': x_t, 'icd': y_t.float()}
        x_next = torch.from_numpy(pair_data[1]).float()
        y_next = torch.from_numpy(pair_labels[1]).long()
        x_next = x_next.transpose(0, 1)

        # # 3. Compute Delta T
        # t0 = self.f_wave['timestamp'][real_idx]
        # t1 = self.f_wave['timestamp'][real_idx + 1]
        # dt_val = (t1 - t0) / self.time_scalar
        # delta_t = torch.tensor([dt_val], dtype=torch.float32)

        # 4. Compute Action Vector (Difference)
        # y is int8, we want float for the network input
        # Values will be -1.0, 0.0, 1.0
        action = (y_next - y_t).float()
        
        # 5. Metadata (Optional, useful for debugging)
        # study_id_t = self.f_wave['study_id'][real_idx]
        # study_id_next = self.f_wave['study_id'][real_idx + 1]
        # is_same_stay = (study_id_t == study_id_next)

        return {'waveform': x_t, 'waveform_next': x_next, 'action': action, 
                'icd': y_t.float(),  # for online probing
                }

    def get_weights_for_balanced_sampling(self):
        """
        Returns a weight vector for WeightedRandomSampler.
        Upsamples 'Changed' pairs to match 'Stable' pairs.
        """
        if not self.return_pairs:
            return None
            
        weights = np.zeros(len(self.valid_indices))
        
        # Assign weights based on class counts
        n_stable = len(self.stable_indices)
        n_changed = len(self.changed_indices)
        
        # Weight = Total / (N_class)
        # But simply: we want probability of changed to be higher.
        # Let's target 50/50 probability.
        
        w_stable = 1.0 / n_stable
        w_changed = 1.0 / n_changed
        
        # We need to map back from real indices to the dataset indices 0..N
        # This is a bit tricky since valid_indices is a list.
        # We can create a boolean mask for the dataset length.
        
        # Efficient way:
        # 1. Create a map of "Real Index" -> "Dataset Index"
        real_to_dataset = {real_idx: i for i, real_idx in enumerate(self.valid_indices)}
        
        for idx in self.stable_indices:
            weights[real_to_dataset[idx]] = w_stable
            
        for idx in self.changed_indices:
            weights[real_to_dataset[idx]] = w_changed
            
        return torch.DoubleTensor(weights)