from __future__ import annotations
import math
from typing import Iterable
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import runtime
from gtae.data import TextAttributedGraph
from gtae.models import GraphEncoder
from attacks import PaperLexicalAttack

class TextFeatureEncoder(nn.Module):

    def __init__(self, name: str, max_length: int=256, dtype: torch.dtype=torch.float32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.encoder = AutoModel.from_pretrained(name, torch_dtype=dtype).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.dim = int(self.encoder.config.hidden_size)
        self.max_length = max_length
        self.cache: dict[str, Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int=64) -> Tensor:
        device = next(self.encoder.parameters()).device
        requested = dict.fromkeys(texts)
        missing = [t for t in requested if t not in self.cache]
        if len(self.cache) + len(missing) > 400000:
            for key in [k for k in self.cache if k not in requested]:
                del self.cache[key]
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            tokens = self.tokenizer(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
            hidden = self.encoder(**tokens).last_hidden_state.float()
            mask = tokens['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            for (offset, text) in enumerate(chunk):
                self.cache[text] = pooled[offset].clone()
        return torch.stack([self.cache[t] for t in texts], dim=0)

class GraphPromptModel(nn.Module):

    def __init__(self, num_classes: int, backbone: str, phi_name: str='sentence-transformers/all-MiniLM-L6-v2', fusion: str='llaga', hidden_dim: int=1024, graph_layers: int=4, graph_heads: int=4, dropout: float=0.1, max_length: int=256, label_texts: list[str] | None=None, neighbours: int=10, translator_tokens: int=4, text_dtype: torch.dtype=torch.bfloat16):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.lm = AutoModelForCausalLM.from_pretrained(backbone, torch_dtype=text_dtype).eval()
        for parameter in self.lm.parameters():
            parameter.requires_grad = False
        self.phi = TextFeatureEncoder(phi_name, max_length=max_length)
        self.text_dim = int(self.lm.config.hidden_size)
        self.feature_dim = self.phi.dim
        self.max_length = max_length
        self.fusion = fusion.lower()
        self.num_classes = num_classes
        self.neighbours = neighbours
        self.translator_tokens = translator_tokens
        self.label_texts = [t.replace('_', ' ') for t in label_texts or [f'class {i}' for i in range(num_classes)]]
        self.graph_encoder = GraphEncoder(self.feature_dim, hidden_dim, graph_layers, graph_heads, dropout)
        self.projector = nn.Sequential(nn.Linear(hidden_dim, self.text_dim), nn.GELU(), nn.Linear(self.text_dim, self.text_dim))
        self.prompt_gate = nn.Sequential(nn.Linear(self.text_dim * 2, self.text_dim), nn.Sigmoid())
        self.translator = nn.Sequential(nn.Linear(self.text_dim, self.text_dim), nn.GELU(), nn.Linear(self.text_dim, self.text_dim * translator_tokens))
        self.pad_node = nn.Parameter(torch.zeros(hidden_dim))
        (self._label_ids, self._label_mask) = self._tokenise_labels()
        self._union_cache: dict[tuple[int, int, int], Tensor] = {}

    def _tokenise_labels(self) -> tuple[Tensor, Tensor]:
        pieces = [self.tokenizer.encode(' ' + label, add_special_tokens=False) for label in self.label_texts]
        width = max((len(p) for p in pieces))
        ids = torch.full((len(pieces), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(pieces), width), dtype=torch.long)
        for (row, piece) in enumerate(pieces):
            ids[row, :len(piece)] = torch.tensor(piece)
            mask[row, :len(piece)] = 1
        return (ids, mask)

    @property
    def device(self) -> torch.device:
        return self.projector[0].weight.device

    def option_block(self) -> str:
        return ', '.join(self.label_texts)

    def build_prompt(self, text: str) -> str:
        return f'You are given a node of a text-attributed graph together with the states of its neighbours.\nNode text: {text}\nChoose the category of this node from: {self.option_block()}.\nCategory:'

    def neighbour_ids(self, graph: TextAttributedGraph, node: int, edge_index: Tensor | None=None) -> Tensor:
        structure = graph.edge_index if edge_index is None else edge_index
        neighbours = structure[1, structure[0] == node]
        neighbours = neighbours[neighbours != node]
        if neighbours.numel() > self.neighbours:
            neighbours = neighbours[::max(1, neighbours.numel() // self.neighbours)][:self.neighbours]
        pad = max(0, self.neighbours - int(neighbours.numel()))
        return torch.cat([torch.tensor([node], device=structure.device), neighbours.long(), torch.full((pad,), -1, dtype=torch.long, device=structure.device)])

    def graph_tokens(self, graph_hidden: Tensor, ids: Tensor) -> Tensor:
        states = torch.where((ids >= 0).unsqueeze(-1), graph_hidden[ids.clamp_min(0)], self.pad_node.unsqueeze(0))
        if self.fusion == 'llaga':
            return self.projector(states)
        own = self.projector(states[:1])
        if self.fusion == 'graphprompter':
            summary = self.projector(states[1:].mean(dim=0, keepdim=True))
            return own + self.prompt_gate(torch.cat([own, summary], dim=-1)) * summary
        if self.fusion == 'graphtranslator':
            return self.translator(own).view(self.translator_tokens, self.text_dim)
        raise ValueError(f'Unsupported fusion: {self.fusion}')

    def node_features(self, texts: list[str]) -> Tensor:
        return self.phi.encode(texts)

    def _union_edges(self, edge_index: Tensor, num_nodes: int, copies: int) -> Tensor:
        key = (int(edge_index.data_ptr()), num_nodes, copies)
        cached = self._union_cache.get(key)
        if cached is None:
            offsets = (torch.arange(copies, device=edge_index.device) * num_nodes).view(copies, 1, 1)
            cached = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1).contiguous()
            if len(self._union_cache) > 8:
                self._union_cache.clear()
            self._union_cache[key] = cached
        return cached

    def graph_hidden_variants(self, base_x: Tensor, edge_index: Tensor, node: int, rows: Tensor) -> Tensor:
        (copies, num_nodes) = (rows.size(0), base_x.size(0))
        x = base_x.unsqueeze(0).repeat(copies, 1, 1)
        x[:, node] = rows.to(x.dtype)
        hidden = self.graph_encoder(x.view(copies * num_nodes, -1), self._union_edges(edge_index, num_nodes, copies))
        return hidden.view(copies, num_nodes, -1)

    def _prompt_batch(self, graph_tokens: Tensor, texts: list[str]):
        table = self.lm.get_input_embeddings().weight
        pieces = [self.tokenizer(self.build_prompt(text), truncation=True, max_length=self.max_length, add_special_tokens=True)['input_ids'] for text in texts]
        width = max((len(p) for p in pieces))
        graph_len = graph_tokens.size(-2)
        (embeds, masks) = ([], [])
        for (row, piece) in enumerate(pieces):
            pad = width - len(piece)
            ids = torch.tensor([self.tokenizer.pad_token_id] * pad + piece, device=table.device)
            text_embeds = table[ids]
            tokens = graph_tokens[row] if graph_tokens.dim() == 3 else graph_tokens
            embeds.append(torch.cat([text_embeds[:pad], tokens.to(text_embeds.dtype), text_embeds[pad:]], dim=0))
            masks.append(torch.tensor([0] * pad + [1] * (graph_len + len(piece)), device=table.device))
        return (torch.stack(embeds), torch.stack(masks))

    def class_logprobs(self, graph_tokens: Tensor, texts: list[str]) -> Tensor:
        (inputs, mask) = self._prompt_batch(graph_tokens, texts)
        positions = (mask.cumsum(dim=-1) - 1).clamp_min(0)
        out = self.lm(inputs_embeds=inputs, attention_mask=mask, position_ids=positions, use_cache=True)
        first = out.logits[:, -1].float().log_softmax(dim=-1)
        (batch, classes) = (len(texts), self.num_classes)
        label_ids = self._label_ids.to(inputs.device)
        label_mask = self._label_mask.to(inputs.device)
        if label_ids.size(1) == 1:
            return first[:, label_ids[:, 0]]
        cache = _expand_cache(out.past_key_values, classes)
        ids = label_ids.repeat(batch, 1)
        lmask = label_mask.repeat(batch, 1)
        prefix_mask = mask.repeat_interleave(classes, dim=0)
        prompt_len = prefix_mask.sum(dim=-1, keepdim=True)
        cont_positions = prompt_len + torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        cont = self.lm(input_ids=ids, attention_mask=torch.cat([prefix_mask, lmask], dim=-1), position_ids=cont_positions, past_key_values=cache, use_cache=False)
        step = cont.logits.float().log_softmax(dim=-1)
        gathered = step[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        head = first.repeat_interleave(classes, dim=0).gather(-1, ids[:, :1]).squeeze(-1)
        tail_mask = lmask[:, 1:].float()
        total = head + (gathered * tail_mask).sum(dim=-1)
        return (total / lmask.sum(dim=-1).clamp_min(1).float()).view(batch, classes)

    def forward(self, graph: TextAttributedGraph, texts: list[str] | None=None, edge_index: Tensor | None=None, x_delta: Tensor | None=None, nodes: Iterable[int] | None=None, batch_size: int=4) -> Tensor:
        values = graph.texts if texts is None else texts
        x = self.node_features(values)
        if x_delta is not None:
            x = x + x_delta
        structure = graph.edge_index if edge_index is None else edge_index
        graph_hidden = self.graph_encoder(x, structure)
        targets = list(range(graph.num_nodes)) if nodes is None else list(nodes)
        outputs = []
        for start in range(0, len(targets), batch_size):
            chunk = targets[start:start + batch_size]
            tokens = torch.stack([self.graph_tokens(graph_hidden, self.neighbour_ids(graph, n, structure)) for n in chunk])
            outputs.append(self.class_logprobs(tokens, [values[n] for n in chunk]))
        return torch.cat(outputs, dim=0)

def _expand_cache(past, repeats: int):
    from transformers import DynamicCache
    legacy = past.to_legacy_cache() if hasattr(past, 'to_legacy_cache') else past
    expanded = tuple(((k.repeat_interleave(repeats, dim=0), v.repeat_interleave(repeats, dim=0)) for (k, v) in legacy))
    return DynamicCache.from_legacy_cache(expanded)

class FlowLexicalAttack(PaperLexicalAttack):

    def __init__(self, *args, synonym_source=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.synonym_source = synonym_source
        self._graph = None
        self._base_x = None
        self._ids = None

    def _synonyms(self, word: str, tag: str) -> list[str]:
        key = (word, tag)
        if key not in self._synonym_cache:
            self._synonym_cache[key] = self.synonym_source.synonyms(word, tag) if self.synonym_source else super()._synonyms(word, tag)
        return self._synonym_cache[key]

    def _word_embeds(self, model, words: list[str]) -> Tensor:
        table = model.lm.get_input_embeddings().weight
        ids = []
        for word in words:
            token = model.tokenizer.encode(word, add_special_tokens=False)
            ids.append(token[0] if token else model.tokenizer.unk_token_id)
        return table[torch.tensor(ids, device=table.device)].float()

    def _semantic_penalty(self, model, word: str, synonym: str) -> Tensor:
        key = (word, synonym)
        if key not in self._penalty_cache:
            pair = self._word_embeds(model, [word, synonym])
            self._penalty_cache[key] = self._semantic_distance(pair[:1], pair[1:]).detach()
        return self._penalty_cache[key]

    def _margins(self, model, node: int, texts: list[str], label: int) -> Tensor:
        graph = self._graph
        out = []
        for start in range(0, len(texts), self.encode_batch):
            chunk = texts[start:start + self.encode_batch]
            rows = model.node_features(chunk)
            hidden = model.graph_hidden_variants(self._base_x, graph.edge_index, node, rows)
            tokens = torch.stack([model.graph_tokens(hidden[i], self._ids) for i in range(len(chunk))])
            out.append(model.class_logprobs(tokens, chunk))
        logits = torch.cat(out, dim=0)
        other = logits.clone()
        other[:, label] = -torch.inf
        return other.max(dim=-1).values - logits[:, label]

    @torch.no_grad()
    def attack_node(self, model, graph: TextAttributedGraph, node: int) -> str:
        import nltk
        self._graph = graph
        if self._base_x is None:
            self._base_x = model.node_features(graph.texts)
        self._ids = model.neighbour_ids(graph, node)
        self.calls += 1
        tokens = self._tokens(graph.texts[node])
        tagged = nltk.pos_tag(tokens)
        replaceable = [(index, token, tag) for (index, (token, tag)) in enumerate(tagged) if tag in self.allowed_pos and len(token) > 2 and token.isalpha()]
        if not replaceable:
            return graph.texts[node]
        budget = max(1, int(math.ceil(len(replaceable) * self.budget_ratio)))
        label = int(graph.y[node])
        working = list(tokens)
        (selected, originals, adversarial, candidate_sets) = ([], [], [], [])
        for _ in range(budget):
            (trials, texts) = ([], [])
            for (index, word, tag) in replaceable:
                if index in selected:
                    continue
                for synonym in self._synonyms(word, tag):
                    trial = list(working)
                    trial[index] = synonym
                    trials.append((index, word, synonym, self._synonyms(word, tag)))
                    texts.append(self._detokenize(trial))
            if not trials:
                break
            self.candidate_evaluations += len(trials)
            margins = self._margins(model, node, texts, label)
            penalties = torch.stack([self._semantic_penalty(model, w, s) for (_, w, s, _) in trials]).to(margins.device)
            scores = margins - self.semantic_weight * penalties
            (index, original, synonym, synonyms) = trials[runtime._first_argmax(scores)]
            working[index] = synonym
            selected.append(index)
            originals.append(original)
            adversarial.append(synonym)
            candidate_sets.append(synonyms)
        refined = self._refine_flow(model, node, label, working, selected, originals, adversarial, candidate_sets)
        for (index, value) in zip(selected, refined):
            working[index] = value
        return self._detokenize(working)

    def _refine_flow(self, model, node, label, working, selected, originals, adversarial, candidate_sets):
        if not originals:
            return adversarial
        original = self._word_embeds(model, originals)
        theta = (self._word_embeds(model, adversarial) - original).clone()

        def project(row: Tensor) -> list[str]:
            targets = original + row
            out = []
            for (target, current, candidates) in zip(targets, adversarial, candidate_sets):
                options = [current] + [c for c in candidates if c != current]
                embeddings = self._word_embeds(model, options)
                similarity = torch.nn.functional.cosine_similarity(target.unsqueeze(0), embeddings, dim=-1)
                out.append(options[int(similarity.argmax())])
            return out

        def objective(rows: Tensor) -> Tensor:
            (texts, drifts) = ([], [])
            for row in rows:
                candidate = list(working)
                for (index, value) in zip(selected, project(row)):
                    candidate[index] = value
                texts.append(self._detokenize(candidate))
                drifts.append(1 - torch.nn.functional.cosine_similarity(original, original + row, dim=-1).mean())
            margins = self._margins(model, node, texts, label)
            return margins - self.semantic_weight * torch.stack(drifts).to(margins.device)
        before = float(objective(theta.unsqueeze(0))[0])
        for _ in range(self.refinement_steps):
            base = objective(theta.unsqueeze(0))[0]
            directions = torch.randn(self.refinement_samples, *theta.shape, device=theta.device)
            directions = directions / directions.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
            shifted = objective(theta.unsqueeze(0) + self.delta * directions)
            gradient = (((shifted - base) / self.delta).view(-1, 1, 1) * directions).mean(dim=0)
            theta = theta + self.learning_rate * gradient - self.learning_rate * self.l1_weight * theta.sign()
        self.refinement_gain.append(float(objective(theta.unsqueeze(0))[0]) - before)
        return project(theta)

    @torch.no_grad()
    def __call__(self, model, graph: TextAttributedGraph, nodes: list[int]):
        self._base_x = model.node_features(graph.texts)
        attacked = graph.clone()
        changes = {}
        try:
            for node in nodes:
                value = self.attack_node(model, attacked, node)
                if value != attacked.texts[node]:
                    attacked.texts[node] = value
                    changes[node] = value
                    self._base_x = self._base_x.clone()
                    self._base_x[node] = model.node_features([value])[0]
        finally:
            self._base_x = None
            self._graph = None
        return (attacked, changes)
