import numpy as np
import torch
from torch.utils.data import Dataset

import lance


class CPSCLanceDataset(Dataset):
    """
    Dataset for CPSC2018 (China Physiological Signal Challenge 2018) finetuning/evaluation.

    Unlike CSN (one row per record), CPSC2018 records vary 6-144s in duration;
    scripts/build_lance_cpsc.py splits each into non-overlapping 5000-sample
    windows, so a single row here is one *window*, not one full recording —
    multiple rows can share the same record_id. Fold assignment happens per
    record and is propagated to all of a record's windows (see the build
    script), so this class's split logic needs no changes for that: filtering
    by fold still keeps every window of a given record in the same split.

    Returns: {"waveform": FloatTensor(12, 5000), "label": FloatTensor(N,)}
    where N is the number of SNOMED-CT Dx classes (see dx_vocabulary_cpsc.json).

    Splits
    ------
    train : folds 0-17
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
        train_frac: float = 1.0,
        cache: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        self._lance_path = lance_path
        self._ds = None  # opened lazily per worker to avoid fork-safety issues
        self._waveforms = None
        self._labels = None

        print(f"reading dataset at {lance_path}")
        _ds_init = lance.dataset(lance_path)
        self._n_classes = _ds_init.schema.field("dx").type.list_size

        meta = _ds_init.to_table(columns=["fold"]).to_pandas()
        meta["lance_idx"] = meta.index

        if split == "train":
            meta = meta[meta["fold"] <= 17].copy()
        elif split == "val":
            meta = meta[meta["fold"] == 18].copy()
        else:
            meta = meta[meta["fold"] == 19].copy()

        if split == "train" and train_frac < 1.0:
            n = max(1, int(len(meta) * train_frac))
            meta = meta.sample(n=n, random_state=0)

        self.rows = np.sort(meta["lance_idx"].values)

        if cache:
            table = _ds_init.take(self.rows.tolist(), columns=["waveform", "dx"])
            self._waveforms = (
                table.column("waveform").combine_chunks().flatten()
                .to_numpy(zero_copy_only=False)
                .reshape(len(self.rows), 5000, 12)
            )
            self._labels = (
                table.column("dx").combine_chunks().flatten()
                .to_numpy(zero_copy_only=False)
                .reshape(len(self.rows), self._n_classes)
            )

    def _get_ds(self):
        if self._ds is None:
            self._ds = lance.dataset(self._lance_path)
        return self._ds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        return self._getitem_single(i)

    def __getitems__(self, indices: list[int]):
        """
        Batch fetch called by DataLoader (PyTorch >= 2.0) instead of N __getitem__ calls.
        Replaces N take([single_idx]) with one take(N_indices), which lets Lance
        group reads by fragment and coalesce I/O — critical for random-access workloads.
        """
        return self._getitems_single(indices)

    def _getitems_single(self, indices: list[int]) -> list[dict]:
        if self._waveforms is not None:
            return [self._getitem_single(i) for i in indices]

        n = len(indices)
        row_indices = [int(self.rows[i]) for i in indices]
        table = self._get_ds().take(row_indices, columns=["waveform", "dx"])

        waveforms = (table.column("waveform").combine_chunks().flatten()
                     .to_numpy(zero_copy_only=False)
                     .reshape(n, 5000, 12)
                     .astype(np.float16))
        labels = (table.column("dx").combine_chunks().flatten()
                  .to_numpy(zero_copy_only=False)
                  .reshape(n, self._n_classes)
                  .astype(np.int8))

        xs = torch.from_numpy(waveforms).float().permute(0, 2, 1)  # (N, 12, 5000)
        ys = torch.from_numpy(labels.astype(np.float32))  # (N, n_classes)

        return [{"waveform": xs[j], "label": ys[j]} for j in range(n)]

    def _getitem_single(self, i: int):
        if self._waveforms is not None:
            x = torch.from_numpy(self._waveforms[i].copy()).float().T
            y = torch.from_numpy(self._labels[i].copy())
        else:
            idx = int(self.rows[i])
            row = self._get_ds().take([idx], columns=["waveform", "dx"])
            x = (row.column("waveform").combine_chunks().slice(0, 1)
                 .flatten().to_numpy(zero_copy_only=False)
                 .reshape(5000, 12).astype(np.float16))
            y = (row.column("dx").combine_chunks().slice(0, 1)
                 .flatten().to_numpy(zero_copy_only=False)
                 .astype(np.int8))
            x = torch.from_numpy(x.copy()).float().T
            y = torch.from_numpy(y.copy())
        return {"waveform": x, "label": y.float()}
