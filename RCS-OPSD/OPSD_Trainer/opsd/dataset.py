import json
from pathlib import Path


class OPSDJsonlDataset:
    """Map-style access to a JSONL file without loading its 313 MB into memory."""

    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open("rb") as handle:
            while handle.readline():
                self.offsets.append(handle.tell())
        self.offsets.insert(0, 0)
        self.offsets.pop()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


class IndexedDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

