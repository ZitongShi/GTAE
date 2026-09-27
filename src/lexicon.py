from __future__ import annotations
import zipfile
from pathlib import Path
import torch

class EmbeddingSynonyms:

    def __init__(self, path: str | Path='data/emb/counter-fitted-vectors.txt', top_k: int=8, threshold: float=0.5, device: str | torch.device='cpu'):
        self.top_k = top_k
        self.threshold = threshold
        (words, rows) = self._load(Path(path))
        self.words = words
        self.index = {word: i for (i, word) in enumerate(words)}
        matrix = torch.tensor(rows, dtype=torch.float32, device=device)
        self.matrix = matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self._cache: dict[str, list[str]] = {}

    @staticmethod
    def _load(path: Path):
        if not path.exists():
            archive = path.with_suffix(path.suffix + '.zip')
            if not archive.exists():
                archive = path.parent / 'cf.zip'
            with zipfile.ZipFile(archive) as zf:
                name = next((n for n in zf.namelist() if n.endswith('.txt')))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(zf.read(name))
        (words, rows) = ([], [])
        with path.open('r', encoding='utf-8', errors='replace') as stream:
            for line in stream:
                parts = line.rstrip().split(' ')
                if len(parts) < 10:
                    continue
                words.append(parts[0])
                rows.append([float(v) for v in parts[1:]])
        return (words, rows)

    def _match_case(self, source: str, candidate: str) -> str:
        if source.isupper():
            return candidate.upper()
        if source[:1].isupper():
            return candidate.capitalize()
        return candidate

    def synonyms(self, word: str, tag: str | None=None) -> list[str]:
        key = word.lower()
        if key in self._cache:
            return [self._match_case(word, w) for w in self._cache[key]]
        row = self.index.get(key)
        if row is None:
            self._cache[key] = []
            return []
        similarity = self.matrix @ self.matrix[row]
        similarity[row] = -1.0
        (values, order) = similarity.topk(min(self.top_k * 3, similarity.numel()))
        picked = [self.words[int(i)] for (value, i) in zip(values.tolist(), order.tolist()) if value >= self.threshold and self.words[int(i)].isalpha()][:self.top_k]
        self._cache[key] = picked
        return [self._match_case(word, w) for w in picked]
