import numpy as np
import torch
from torch.utils.data import Dataset

import lance


class MIMICLanceDataset(Dataset):
    """
    Unified dataset for pre-training and evaluation on MIMIC-IV-ECG.

    mode="triage"
        First ECG of each hospital stay (ecg_no_within_stay == 0).
        Returns: {"waveform": FloatTensor(12, 5000), "label": FloatTensor(76,)}

    mode="monitoring"
        All ECGs in the split.
        Returns: {"waveform": FloatTensor(12, 5000), "label": FloatTensor(76,)}

    mode="pair"
        Longitudinal pairs (Xt, Xt+1) with pathology transition vector.
        Requires pairs_path. Returns: (xt, xt1, yt, at) as FloatTensors,
        waveforms shaped (12, 5000).

    Splits
    ------
    train : folds 0–17
    val   : fold 18
    test  : fold 19

    DataLoader usage
    ----------------
    Lance is not fork-safe. Always use multiprocessing_context="spawn":
        DataLoader(ds, num_workers=N, multiprocessing_context="spawn")
    The lance handle is opened lazily in each worker to avoid pickling it.
    """

    def __init__(
        self,
        lance_path: str,
        split: str = "train",
        mode: str = "triage",
        pairs_path: str | None = None,
        pair_types: tuple[str, ...] = ("within_stay", "cross_stay"),
        train_frac: float = 1.0,
        cache: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        assert mode in ("triage", "monitoring", "pair"), f"Unknown mode: {mode!r}"
        if mode == "pair" and pairs_path is None:
            raise ValueError("pairs_path is required when mode='pair'")

        self._lance_path = lance_path
        self._mode = mode
        self._ds = None  # opened lazily per worker to avoid fork-safety issues
        self._waveforms = None
        self._labels = None
        self._idx_to_pos = None  # used in pair mode only

        _ds_init = lance.dataset(lance_path)

        if mode == "pair":
            type_filter = " OR ".join(f"pair_type = '{t}'" for t in pair_types)
            fold_filter = "fold <= 17" if split == "train" else ("fold = 18" if split == "val" else "fold = 19")

            pairs_df = (
                lance.dataset(pairs_path)
                .to_table(filter=f"({fold_filter}) AND ({type_filter})")
                .to_pandas()[["idx_t", "idx_t1", "subject_id"]]
            )

            if split == "train" and train_frac < 1.0:
                rng = np.random.default_rng(0)
                patients = pairs_df["subject_id"].unique()
                n = max(1, int(len(patients) * train_frac))
                selected = rng.choice(patients, size=n, replace=False)
                pairs_df = pairs_df[pairs_df["subject_id"].isin(selected)]

            self.pairs = pairs_df[["idx_t", "idx_t1"]].values  # int64 ndarray (N, 2)

            if cache:
                unique_idxs = np.unique(self.pairs)
                table = _ds_init.take(unique_idxs.tolist(), columns=["waveform", "icd"])
                self._waveforms = (
                    table.column("waveform").combine_chunks().flatten()
                    .to_numpy(zero_copy_only=False)
                    .reshape(len(unique_idxs), 5000, 12)
                )
                self._labels = (
                    table.column("icd").combine_chunks().flatten()
                    .to_numpy(zero_copy_only=False)
                    .reshape(len(unique_idxs), 76)
                )
                self._idx_to_pos = {int(idx): pos for pos, idx in enumerate(unique_idxs)}

        else:
            meta = _ds_init.to_table(
                columns=["subject_id", "ecg_time", "fold", "ecg_no_within_stay"]
            ).to_pandas()
            meta["lance_idx"] = meta.index

            if split == "train":
                meta = meta[meta["fold"] <= 17].copy()
            elif split == "val":
                meta = meta[meta["fold"] == 18].copy()
            else:
                meta = meta[meta["fold"] == 19].copy()

            if split == "train" and train_frac < 1.0:
                rng = np.random.default_rng(0)
                patients = meta["subject_id"].unique()
                n = max(1, int(len(patients) * train_frac))
                selected = rng.choice(patients, size=n, replace=False)
                meta = meta[meta["subject_id"].isin(selected)].copy()

            if mode == "triage":
                self.rows = meta.loc[meta["ecg_no_within_stay"] == 0, "lance_idx"].values
            else:
                self.rows = meta["lance_idx"].values

            self.rows = np.sort(self.rows)

            if cache:
                table = _ds_init.take(self.rows.tolist(), columns=["waveform", "icd"])
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
        if self._mode == "pair":
            return len(self.pairs)
        return len(self.rows)

    def __getitem__(self, i: int):
        if self._mode == "pair":
            return self._getitem_pair(i)
        return self._getitem_single(i)

    def _getitem_single(self, i: int):
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
        return {"waveform": x, "label": y.float()}

    def _getitem_pair(self, i: int):
        idx_t, idx_t1 = int(self.pairs[i, 0]), int(self.pairs[i, 1])

        if self._waveforms is not None:
            xt  = self._waveforms[self._idx_to_pos[idx_t]]
            xt1 = self._waveforms[self._idx_to_pos[idx_t1]]
            yt  = self._labels[self._idx_to_pos[idx_t]]
            yt1 = self._labels[self._idx_to_pos[idx_t1]]
        else:
            rows = self._get_ds().take([idx_t, idx_t1], columns=["waveform", "icd"])
            xt  = _fixed_list_to_ndarray(rows, "waveform", 0, np.float16).reshape(5000, 12)
            xt1 = _fixed_list_to_ndarray(rows, "waveform", 1, np.float16).reshape(5000, 12)
            yt  = _fixed_list_to_ndarray(rows, "icd", 0, np.int8)
            yt1 = _fixed_list_to_ndarray(rows, "icd", 1, np.int8)

        at = np.clip(yt1.astype(np.int16) - yt.astype(np.int16), -1, 1).astype(np.int8)

        return (
            torch.from_numpy(xt.copy()).float().T,
            torch.from_numpy(xt1.copy()).float().T,
            torch.from_numpy(yt.copy()),
            torch.from_numpy(at.copy()),
        )


def _fixed_list_to_ndarray(table, column: str, row: int, dtype) -> np.ndarray:
    return np.array(table.column(column)[row].as_py(), dtype=dtype)
