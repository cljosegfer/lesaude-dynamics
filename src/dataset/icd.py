import json
from pathlib import Path

import numpy as np
import pyarrow as pa


class ICDVocabulary:
    def __init__(self, path):
        with open(path) as f:
            self.codes = json.load(f)
        self.code_to_idx = {c: i for i, c in enumerate(self.codes)}

    def __len__(self):
        return len(self.codes)

    def __repr__(self):
        return f"ICDVocabulary({len(self)} codes)"

    def encode(self, raw_codes: list[str]) -> np.ndarray:
        """list[str] of ICD codes → int8 binary vector of shape (N,)."""
        vec = np.zeros(len(self), dtype=np.int8)
        for c in raw_codes:
            if c in self.code_to_idx:
                vec[self.code_to_idx[c]] = 1
        return vec

    def decode(self, vec: np.ndarray) -> list[str]:
        """int8 binary vector → list of ICD code strings."""
        return [self.codes[i] for i, v in enumerate(vec) if v]

    @classmethod
    def from_hdf5(cls, hdf5_vocabulary, path: str | Path) -> "ICDVocabulary":
        """Save vocabulary from an HDF5 object-dtype array to JSON, then load it."""
        codes = [v.decode() if isinstance(v, bytes) else str(v) for v in hdf5_vocabulary]
        path = Path(path)
        with open(path, "w") as f:
            json.dump(codes, f)
        return cls(path)

    def add_lance_column(self, lance_path: str | Path, column_name: str) -> None:
        """
        Derive a new binary ICD column from icd_raw in an existing Lance dataset.
        Only the new column fragment is written; waveform data is untouched.

        Example — adding a 2-digit grouping:
            vocab_2digit = ICDVocabulary("icd_vocabulary_2digit.json")
            vocab_2digit.add_lance_column("mimic_iv_ecg.lance", "icd_2digit")
        """
        import lance

        codes = self.codes
        n = len(codes)
        code_to_idx = self.code_to_idx

        def recode(batch: pa.RecordBatch) -> pa.Array:
            raw_lists = batch["icd_raw"].to_pylist()
            result = []
            for raw in raw_lists:
                vec = [0] * n
                for c in raw:
                    if c in code_to_idx:
                        vec[code_to_idx[c]] = 1
                result.append(vec)
            return pa.array(result, type=pa.list_(pa.int8(), n))

        ds = lance.dataset(str(lance_path))
        ds.add_columns({column_name: recode}, read_columns=["icd_raw"])
