from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from transformers import AutoModel, AutoTokenizer

from .data import TextAttributedGraph


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for layer in range(layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        hidden = x
        for convolution, norm in zip(self.layers, self.norms):
            hidden = convolution(hidden, edge_index)
            hidden = norm(hidden)
            hidden = F.gelu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden


class GraphTextModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        backbone: str,
        fusion: str = "llaga",
        hidden_dim: int = 1024,
        graph_layers: int = 4,
        graph_heads: int = 4,
        dropout: float = 0.1,
        freeze_text_encoder: bool = True,
        max_length: int = 256,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.text_encoder = AutoModel.from_pretrained(backbone)
        self.text_dim = int(self.text_encoder.config.hidden_size)
        self.max_length = max_length
        self.fusion = fusion.lower()
        self.graph_encoder = GraphEncoder(input_dim, hidden_dim, graph_layers, graph_heads, dropout)
        self.graph_projector = nn.Sequential(
            nn.Linear(hidden_dim, self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        )
        self.translator = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.GELU(),
            nn.LayerNorm(self.text_dim),
        )
        self.prompt_gate = nn.Sequential(nn.Linear(self.text_dim * 2, self.text_dim), nn.Sigmoid())
        classifier_dim = self.text_dim * 2 if self.fusion == "llaga" else self.text_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )
        if freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> Tensor:
        outputs = []
        grad_enabled = any(parameter.requires_grad for parameter in self.text_encoder.parameters())
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                encoded = self.text_encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    def fuse(self, graph_hidden: Tensor, text_hidden: Tensor) -> Tensor:
        graph_hidden = self.graph_projector(graph_hidden)
        if self.fusion == "llaga":
            return torch.cat([graph_hidden, text_hidden], dim=-1)
        if self.fusion == "graphprompter":
            gate = self.prompt_gate(torch.cat([graph_hidden, text_hidden], dim=-1))
            return text_hidden + gate * graph_hidden
        if self.fusion == "graphtranslator":
            return text_hidden + self.translator(graph_hidden)
        raise ValueError(f"Unsupported fusion: {self.fusion}")

    def forward(
        self,
        graph: TextAttributedGraph,
        texts: list[str] | None = None,
        edge_index: Tensor | None = None,
        x_delta: Tensor | None = None,
    ) -> Tensor:
        x = graph.x if x_delta is None else graph.x + x_delta
        structure = graph.edge_index if edge_index is None else edge_index
        graph_hidden = self.graph_encoder(x, structure)
        text_hidden = self.encode_texts(graph.texts if texts is None else texts)
        return self.classifier(self.fuse(graph_hidden, text_hidden))

    def predict_proba(self, graph: TextAttributedGraph, **kwargs) -> Tensor:
        return self.forward(graph, **kwargs).softmax(dim=-1)

    def token_embeddings(self, tokens: Iterable[str]) -> Tensor:
        embedding = self.text_encoder.get_input_embeddings().weight
        ids = []
        for token in tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            ids.append(token_ids[0] if token_ids else self.tokenizer.unk_token_id)
        return embedding[torch.tensor(ids, device=embedding.device)]
