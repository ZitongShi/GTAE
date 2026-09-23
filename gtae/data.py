from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid


@dataclass
class TextAttributedGraph:
    x: Tensor
    edge_index: Tensor
    y: Tensor
    texts: list[str]
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    global_node_ids: Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item() + 1)

    def to(self, device: str | torch.device) -> "TextAttributedGraph":
        values = {
            "x": self.x.to(device),
            "edge_index": self.edge_index.to(device),
            "y": self.y.to(device),
            "train_mask": self.train_mask.to(device),
            "val_mask": self.val_mask.to(device),
            "test_mask": self.test_mask.to(device),
            "global_node_ids": None if self.global_node_ids is None else self.global_node_ids.to(device),
        }
        return replace(self, **values)

    def clone(self) -> "TextAttributedGraph":
        return TextAttributedGraph(
            x=self.x.clone(),
            edge_index=self.edge_index.clone(),
            y=self.y.clone(),
            texts=list(self.texts),
            train_mask=self.train_mask.clone(),
            val_mask=self.val_mask.clone(),
            test_mask=self.test_mask.clone(),
            global_node_ids=None if self.global_node_ids is None else self.global_node_ids.clone(),
        )


def _mask(size: int, indices: Tensor) -> Tensor:
    value = torch.zeros(size, dtype=torch.bool)
    value[indices.view(-1).long()] = True
    return value


def _default_masks(size: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator)
    train_end = int(size * 0.6)
    val_end = int(size * 0.8)
    return _mask(size, order[:train_end]), _mask(size, order[train_end:val_end]), _mask(size, order[val_end:])


def _texts(data: Data, path: str | None) -> list[str]:
    if path:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, dict):
            value = [value[str(i)] for i in range(data.num_nodes)]
        return [str(item) for item in value]
    if hasattr(data, "texts"):
        return [str(item) for item in data.texts]
    if hasattr(data, "text"):
        return [str(item) for item in data.text]
    return [" ".join(f"f{j}:{v:.6g}" for j, v in enumerate(row.tolist()) if v != 0) for row in data.x]


def _from_data(data: Data, texts_path: str | None, seed: int) -> TextAttributedGraph:
    y = data.y.view(-1).long()
    if hasattr(data, "train_mask") and data.train_mask is not None:
        train_mask = data.train_mask[:, 0] if data.train_mask.ndim > 1 else data.train_mask
        val_mask = data.val_mask[:, 0] if data.val_mask.ndim > 1 else data.val_mask
        test_mask = data.test_mask[:, 0] if data.test_mask.ndim > 1 else data.test_mask
    else:
        train_mask, val_mask, test_mask = _default_masks(data.num_nodes, seed)
    return TextAttributedGraph(
        x=data.x.float(),
        edge_index=data.edge_index.long(),
        y=y,
        texts=_texts(data, texts_path),
        train_mask=train_mask.bool(),
        val_mask=val_mask.bool(),
        test_mask=test_mask.bool(),
        global_node_ids=torch.arange(data.num_nodes),
    )


def _load_processed(path: str) -> Data:
    value: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, Data):
        return value
    if isinstance(value, dict) and "data" in value and isinstance(value["data"], Data):
        data = value["data"]
        if "texts" in value:
            data.texts = value["texts"]
        return data
    if isinstance(value, dict):
        return Data(**value)
    raise TypeError(f"Unsupported processed graph type: {type(value)!r}")


def load_text_attributed_graph(config: dict[str, Any], seed: int = 42) -> TextAttributedGraph:
    name = config["name"].lower()
    root = config.get("root", "data")
    processed_path = config.get("processed_path")
    if processed_path:
        data = _load_processed(processed_path)
    elif name == "cora":
        data = Planetoid(root, "Cora")[0]
    elif name == "pubmed":
        data = Planetoid(root, "PubMed")[0]
    elif name in {"ogbn-arxiv", "arxiv"}:
        dataset = PygNodePropPredDataset("ogbn-arxiv", root)
        data = dataset[0]
        split = dataset.get_idx_split()
        data.train_mask = _mask(data.num_nodes, split["train"])
        data.val_mask = _mask(data.num_nodes, split["valid"])
        data.test_mask = _mask(data.num_nodes, split["test"])
    elif name in {"amz-computers", "amazon-computers", "computers"}:
        data = Amazon(root, "Computers")[0]
    elif name in {"amz-sports", "amazon-sports", "sports"}:
        candidate = Path(root) / "amz-sports" / "processed" / "data.pt"
        if not candidate.exists():
            raise FileNotFoundError("amz-sports requires dataset.processed_path or data/amz-sports/processed/data.pt")
        data = _load_processed(str(candidate))
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return _from_data(data, config.get("texts_path"), seed)
