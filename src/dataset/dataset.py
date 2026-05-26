import numpy as np
import torch
from torch.utils.data import Dataset

import lance


class MIMICLanceDataset(Dataset):
    """
    Pre-training dataset that yields longitudinal ECG pairs (Xt, Xt+1) together
    with their ICD label vectors and the action vector at = clip(yt+1 - yt, -1, 1).

    Each sample is a tuple:
        (xt, xt1, yt, at)
        xt, xt1 : FloatTensor (5000, 12)   — raw 12-lead waveforms
        yt       : int8 Tensor (76,)        — ICD label vector at time t
        at       : int8 Tensor (76,)        — pathology transition vector ∈ {-1, 0, 1}^76

    Pair types
    ----------
    within_stay : consecutive ECGs from the same hospital stay
    cross_stay  : last ECG of stay N → first ECG of stay N+1 for the same patient

    DataLoader usage
    ----------------
    Lance is not fork-safe. Always use multiprocessing_context="spawn":
        DataLoader(ds, num_workers=N, multiprocessing_context="spawn")
    The lance handle is opened lazily in each worker to avoid pickling it.
    """

    def __init__(
        self,
        lance_path: str,
        pairs_path: str,
        fold: int,
        split: str = "train",
        pair_types: tuple[str, ...] = ("within_stay", "cross_stay"),
        train_frac: float = 1.0,
        cache: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        self._lance_path = lance_path
        self._ds = None  # opened lazily per worker to avoid fork-safety issues
        self._waveforms = None  # populated below if cache=True
        self._labels = None
        self._idx_to_pos = None

        type_filter = " OR ".join(f"pair_type = '{t}'" for t in pair_types)
        fold_filter = f"fold != {fold}" if split == "train" else f"fold = {fold}"

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

        self.pairs = pairs_df[["idx_t", "idx_t1"]].values  # int64 ndarray, shape (N, 2)

        if cache:
            _ds = lance.dataset(lance_path)
            unique_idxs = np.unique(self.pairs)  # already sorted by np.unique
            table = _ds.take(unique_idxs.tolist(), columns=["waveform", "icd"])
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

    def _get_ds(self):
        if self._ds is None:
            self._ds = lance.dataset(self._lance_path)
        return self._ds

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
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
            torch.from_numpy(xt.copy()).float(),
            torch.from_numpy(xt1.copy()).float(),
            torch.from_numpy(yt.copy()),
            torch.from_numpy(at.copy()),
        )


def _fixed_list_to_ndarray(table, column: str, row: int, dtype) -> np.ndarray:
    return np.array(table.column(column)[row].as_py(), dtype=dtype)
