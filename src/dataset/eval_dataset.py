import numpy as np
import torch
from torch.utils.data import Dataset

import lance


class MIMICLanceEvalDataset(Dataset):
    """
    Evaluation dataset for the triage and monitoring tasks.

    Each sample is a tuple:
        (x, y)
        x : FloatTensor (5000, 12) — raw 12-lead waveform
        y : int8 Tensor (76,)      — ICD label vector

    Modes
    -----
    triage     : chronologically first ECG per patient (simulates first encounter).
    monitoring : all ECGs for all patients in the split.

    Implementation note
    -------------------
    All metadata rows (no waveforms) are read into memory at init to determine
    which Lance row indices belong to this split/mode. Each __getitem__ then fetches
    exactly one waveform + label row from Lance.

    DataLoader usage
    ----------------
    Lance is not fork-safe. Always use multiprocessing_context="spawn":
        DataLoader(ds, num_workers=N, multiprocessing_context="spawn")
    The lance handle is opened lazily in each worker to avoid pickling it.
    """

    def __init__(
        self,
        lance_path: str,
        fold: int,
        split: str = "val",
        mode: str = "triage",
        train_frac: float = 1.0,
        cache: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        assert mode in ("triage", "monitoring"), f"Unknown mode: {mode!r}"

        self._lance_path = lance_path
        self._ds = None  # opened lazily per worker to avoid fork-safety issues
        self._waveforms = None  # populated below if cache=True
        self._labels = None

        # Read all metadata once — fast, no waveforms involved.
        # Use a local handle; don't store it so workers don't inherit an open handle.
        _ds_init = lance.dataset(lance_path)
        meta = _ds_init.to_table(
            columns=["subject_id", "ecg_time", "fold"]
        ).to_pandas()
        meta["lance_idx"] = meta.index  # position in the full Lance dataset

        if split == "train":
            meta = meta[meta["fold"] != fold].copy()
        else:
            meta = meta[meta["fold"] == fold].copy()

        if split == "train" and train_frac < 1.0:
            rng = np.random.default_rng(0)
            patients = meta["subject_id"].unique()
            n = max(1, int(len(patients) * train_frac))
            selected = rng.choice(patients, size=n, replace=False)
            meta = meta[meta["subject_id"].isin(selected)].copy()

        if mode == "triage":
            # Earliest ECG per patient by ecg_time.
            first_idx = meta.loc[
                meta.groupby("subject_id")["ecg_time"].idxmin(), "lance_idx"
            ]
            self.rows = first_idx.values
        else:
            self.rows = meta["lance_idx"].values

        # Sorted row order benefits both Lance I/O and cache loading.
        self.rows = np.sort(self.rows)

        if cache:
            table = _ds_init.take(self.rows.tolist(), columns=["waveform", "icd"])
            # Use PyArrow buffer path — avoids materialising Python float objects
            # (to_pylist() would expand 8.6 GB of float16 into ~120 GB of Python objects).
            self._waveforms = (
                table.column("waveform").combine_chunks().flatten()
                .to_numpy(zero_copy_only=False)
                .reshape(len(self.rows), 5000, 12)
            )
            self._labels = (
                table.column("icd").combine_chunks().flatten()
                .to_numpy(zero_copy_only=False)
                .reshape(len(self.rows), 76)
            )

    def _get_ds(self):
        if self._ds is None:
            self._ds = lance.dataset(self._lance_path)
        return self._ds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        if self._waveforms is not None:
            x = torch.from_numpy(self._waveforms[i].copy()).float().T
            y = torch.from_numpy(self._labels[i].copy())
        else:
            idx = int(self.rows[i])
            row = self._get_ds().take([idx], columns=["waveform", "icd"])
            x = np.array(row.column("waveform")[0].as_py(), dtype=np.float16).reshape(5000, 12)
            y = np.array(row.column("icd")[0].as_py(), dtype=np.int8)
            x = torch.from_numpy(x.copy()).float().T
            y = torch.from_numpy(y.copy())

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        return {"waveform": x, "label": y.float()}
