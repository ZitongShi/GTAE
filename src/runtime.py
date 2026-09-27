from __future__ import annotations
import copy
import math
from typing import Any, Iterable
import torch
from torch import Tensor
from gtae import attack as up_attack
from gtae import partition as up_partition
from gtae.attack import GTAE, InfluenceGuidedTopologyAttack, LexicalEmbeddingAttack
from gtae.data import TextAttributedGraph
from gtae.defense import STRUM
from gtae.federated import FederatedExperiment
from gtae.models import GraphTextModel

def _first_argmax(values: Tensor) -> int:
    return int((values == values.max()).nonzero(as_tuple=False)[0])

def _margin_from_logits(logits: Tensor, label: int, fixed_target: int | None=None) -> Tensor:
    logp = logits.log_softmax(dim=-1)
    if fixed_target is not None:
        return logp[fixed_target] - logp[label]
    other = logp.clone()
    other[label] = -torch.inf
    return other.max() - logp[label]

def _topology_margin(self: InfluenceGuidedTopologyAttack, model: GraphTextModel, graph: TextAttributedGraph, edge_index: Tensor, node: int) -> Tensor:
    logits = model(graph, edge_index=edge_index)[node]
    return _margin_from_logits(logits, int(graph.y[node]), getattr(self, '_fixed_target', None))

def _aggregation_weights(self: STRUM, scores: list[Tensor]) -> Tensor:
    gamma = torch.stack([torch.as_tensor(score).detach().float().cpu().reshape(()) for score in scores])
    gamma = gamma.clamp_min(0.0)
    total = gamma.sum()
    if float(total) <= 0.0:
        return torch.full_like(gamma, 1.0 / gamma.numel())
    return gamma / total

def _install_text_cache(model: GraphTextModel, slack: int=4096) -> None:
    model._text_cache = {}
    model._text_cache_pinned = set()
    model._text_cache_slack = slack
    model._text_cache_stats = {'hits': 0, 'misses': 0}

    def _encode_uncached(texts: list[str], batch_size: int) -> Tensor:
        outputs = []
        grad_enabled = any((parameter.requires_grad for parameter in model.text_encoder.parameters()))
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            for start in range(0, len(texts), batch_size):
                tokens = model.tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=model.max_length, return_tensors='pt').to(model.device)
                encoded = model.text_encoder(**tokens).last_hidden_state.float()
                mask = tokens['attention_mask'].unsqueeze(-1).float()
                pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    def encode_texts(texts: list[str], batch_size: int=32) -> Tensor:
        if any((parameter.requires_grad for parameter in model.text_encoder.parameters())):
            return _encode_uncached(texts, batch_size)
        cache = model._text_cache
        pinned = model._text_cache_pinned
        requested = dict.fromkeys(texts)
        loose = [key for key in cache if key not in pinned and key not in requested]
        if len(cache) - len(pinned) + len(requested) > model._text_cache_slack:
            for key in loose:
                del cache[key]
        missing = [text for text in requested if text not in cache]
        model._text_cache_stats['hits'] += len(texts) - len(missing)
        model._text_cache_stats['misses'] += len(missing)
        if missing and getattr(model, '_text_cache_offloaded', False):
            raise RuntimeError(f'{len(missing)} uncached text(s) after the LM was offloaded; offloading is only valid when no new text can appear')
        if missing:
            for start in range(0, len(missing), batch_size):
                chunk = missing[start:start + batch_size]
                pooled = _encode_uncached(chunk, batch_size)
                for (offset, text) in enumerate(chunk):
                    cache[text] = pooled[offset].clone()
        return torch.stack([cache[text] for text in texts], dim=0)

    def offload_text_encoder() -> None:
        model.text_encoder.to('cpu')
        model._text_cache_offloaded = True
        torch.cuda.empty_cache()

    def pin_texts(texts: Iterable[str]) -> None:
        values = list(dict.fromkeys(texts))
        for start in range(0, len(values), 256):
            chunk = values[start:start + 256]
            encode_texts(chunk, batch_size=32)
            model._text_cache_pinned.update(chunk)
    model.encode_texts = encode_texts
    model.pin_texts = pin_texts
    model.offload_text_encoder = offload_text_encoder
    model._text_cache_offloaded = False

def _install_fp32_token_embeddings(model: GraphTextModel) -> None:
    original = model.token_embeddings

    def token_embeddings(tokens: Iterable[str]) -> Tensor:
        return original(tokens).float()
    model.token_embeddings = token_embeddings

def _install_state_dict_filter(model: GraphTextModel) -> None:
    keep = tuple((name for (name, _) in model.named_children() if name != 'text_encoder'))
    original_state_dict = model.state_dict
    original_load = model.load_state_dict

    def state_dict(*args: Any, **kwargs: Any):
        return {k: v for (k, v) in original_state_dict(*args, **kwargs).items() if k.startswith(keep)}

    def load_state_dict(state, strict: bool=True, **kwargs: Any):
        return original_load(state, strict=False, **kwargs)
    model.state_dict = state_dict
    model.load_state_dict = load_state_dict
_SKIP_ON_COPY = {'encode_texts', 'pin_texts', 'offload_text_encoder', 'state_dict', 'load_state_dict', 'token_embeddings', '__deepcopy__'}

def _install_shared_deepcopy(model: GraphTextModel) -> None:

    def __deepcopy__(memo: dict) -> GraphTextModel:
        memo[id(model.text_encoder)] = model.text_encoder
        memo[id(model.tokenizer)] = model.tokenizer
        memo[id(model._text_cache)] = model._text_cache
        memo[id(model._text_cache_pinned)] = model._text_cache_pinned
        clone = model.__class__.__new__(model.__class__)
        memo[id(model)] = clone
        for (key, value) in model.__dict__.items():
            if key in _SKIP_ON_COPY:
                continue
            clone.__dict__[key] = copy.deepcopy(value, memo)
        _install_text_cache(clone, slack=model._text_cache_slack)
        clone._text_cache = model._text_cache
        clone._text_cache_pinned = model._text_cache_pinned
        clone._text_cache_stats = model._text_cache_stats
        _install_fp32_token_embeddings(clone)
        _install_state_dict_filter(clone)
        _install_shared_deepcopy(clone)
        return clone
    model.__deepcopy__ = __deepcopy__

def prepare_model(model: GraphTextModel, text_dtype: torch.dtype=torch.bfloat16, slack: int=4096) -> GraphTextModel:
    if not any((parameter.requires_grad for parameter in model.text_encoder.parameters())):
        model.text_encoder.to(text_dtype)
        model.text_encoder.eval()
    if model.tokenizer.padding_side != 'right':
        model.tokenizer.padding_side = 'right'
    _install_text_cache(model, slack=slack)
    _install_fp32_token_embeddings(model)
    _install_state_dict_filter(model)
    _install_shared_deepcopy(model)
    return model

@torch.no_grad()
def _fast_topology_attack_node(self: InfluenceGuidedTopologyAttack, model: GraphTextModel, graph: TextAttributedGraph, node: int) -> tuple[Tensor, list[tuple[int, int]]]:
    device = graph.edge_index.device
    text_row = model.encode_texts([graph.texts[node]])
    label = int(graph.y[node])
    candidates = self._candidates(graph, node)

    def margin(edge_index: Tensor) -> Tensor:
        graph_hidden = model.graph_encoder(graph.x, edge_index)[node:node + 1]
        logits = model.classifier(model.fuse(graph_hidden, text_row))[0]
        return _margin_from_logits(logits, label, getattr(self, '_fixed_target', None))
    pairs = up_attack._edge_pairs(graph.edge_index)
    degree = sum((node in pair for pair in pairs))
    budget = max(1, int(math.ceil(degree * self.budget_scale)))
    flipped: list[tuple[int, int]] = []
    for _ in range(budget):
        best_pair = None
        best_gain = -torch.inf
        current_margin = margin(up_attack._edge_index(pairs, device))
        for candidate in candidates:
            trial_pairs = up_attack._toggle(pairs, node, candidate)
            gain = margin(up_attack._edge_index(trial_pairs, device)) - current_margin
            if gain > best_gain:
                best_gain = gain
                best_pair = (node, candidate)
        if best_pair is None or best_gain <= 0:
            break
        pairs = up_attack._toggle(pairs, *best_pair)
        flipped.append(best_pair)
    return (up_attack._edge_index(pairs, device), flipped)

class FastLexicalEmbeddingAttack(LexicalEmbeddingAttack):

    def __init__(self, *args: Any, encode_batch: int=64, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.encode_batch = int(encode_batch)
        self._synonym_cache: dict[tuple[str, str], list[str]] = {}
        self._penalty_cache: dict[tuple[str, str], Tensor] = {}
        self._graph_hidden: Tensor | None = None
        self.calls = 0
        self.candidate_evaluations = 0
        self.refine_positions = 0
        self.refine_changed = 0
        self.refine_drift_delta = 0.0

    def _synonyms(self, word: str, tag: str) -> list[str]:
        key = (word, tag)
        if key not in self._synonym_cache:
            self._synonym_cache[key] = super()._synonyms(word, tag)
        return self._synonym_cache[key]

    def _refine(self, model, original_words, adversarial_words, candidate_sets):
        refined = super()._refine(model, original_words, adversarial_words, candidate_sets)
        for (original, before, after) in zip(original_words, adversarial_words, refined):
            self.refine_positions += 1
            if before != after:
                self.refine_changed += 1
                self.refine_drift_delta += float(self._semantic_penalty(model, original, after) - self._semantic_penalty(model, original, before))
        return refined

    def _semantic_penalty(self, model: GraphTextModel, word: str, synonym: str) -> Tensor:
        key = (word, synonym)
        if key not in self._penalty_cache:
            original = model.token_embeddings([word])
            replacement = model.token_embeddings([synonym])
            self._penalty_cache[key] = self._semantic_distance(original, replacement).detach()
        return self._penalty_cache[key]

    def _margins(self, model: GraphTextModel, node: int, texts: list[str], label: int) -> Tensor:
        text_hidden = model.encode_texts(texts, batch_size=self.encode_batch)
        graph_row = self._graph_hidden[node:node + 1].expand(len(texts), -1)
        logits = model.classifier(model.fuse(graph_row, text_hidden))
        other = logits.clone()
        other[:, label] = -torch.inf
        return other.max(dim=-1).values - logits[:, label]

    @torch.no_grad()
    def attack_node(self, model: GraphTextModel, graph: TextAttributedGraph, node: int) -> str:
        import nltk
        if self._graph_hidden is None:
            self._graph_hidden = model.graph_encoder(graph.x, graph.edge_index)
        self.calls += 1
        tokens = self._tokens(graph.texts[node])
        tagged = nltk.pos_tag(tokens)
        replaceable = [(index, token, tag) for (index, (token, tag)) in enumerate(tagged) if tag in self.allowed_pos and len(token) > 2 and token.isalpha()]
        if not replaceable:
            return graph.texts[node]
        budget = max(1, int(math.ceil(len(replaceable) * self.budget_ratio)))
        label = int(graph.y[node])
        working = list(tokens)
        selected: list[int] = []
        selected_original: list[str] = []
        selected_adversarial: list[str] = []
        selected_candidates: list[list[str]] = []
        for _ in range(budget):
            trials: list[tuple[int, str, str, list[str]]] = []
            texts: list[str] = []
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
            (index, original, synonym, synonyms) = trials[_first_argmax(scores)]
            working[index] = synonym
            selected.append(index)
            selected_original.append(original)
            selected_adversarial.append(synonym)
            selected_candidates.append(synonyms)
        refined = self._refine(model, selected_original, selected_adversarial, selected_candidates)
        for (index, value) in zip(selected, refined):
            working[index] = value
        return self._detokenize(working)

    @torch.no_grad()
    def __call__(self, model: GraphTextModel, graph: TextAttributedGraph, nodes: list[int]):
        self._graph_hidden = model.graph_encoder(graph.x, graph.edge_index)
        try:
            return super().__call__(model, graph, nodes)
        finally:
            self._graph_hidden = None

class StagedGTAE(GTAE):

    def __init__(self, topology, lexical, stages: str='both'):
        super().__init__(topology, lexical)
        assert stages in {'both', 'structure', 'text'}
        self.stages = stages

    def __call__(self, model: GraphTextModel, graph: TextAttributedGraph, nodes: list[int]):
        model.eval()
        current = graph
        flipped: list[tuple[int, int]] = []
        text_changes: dict[int, str] = {}
        if self.stages in {'both', 'structure'}:
            (current, flipped) = self.topology(model, current, nodes)
        if self.stages in {'both', 'text'}:
            (current, text_changes) = self.lexical(model, current, nodes)
        return up_attack.AttackResult(graph=current, perturbed_nodes=list(nodes), flipped_edges=flipped, text_changes=text_changes)

class CappedSTRUM(STRUM):

    def __init__(self, *args: Any, train_cap: int=32, eval_cap: int=64, seed: int=42, parts: str='both', weighted_aggregation: bool=True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert parts in {'both', 'structure', 'text'}
        self.train_cap = int(train_cap)
        self.eval_cap = int(eval_cap)
        self.seed = int(seed)
        self.parts = parts
        self.weighted_aggregation = bool(weighted_aggregation)
        self.client_sizes: list[float] | None = None
        self._draws = 0

    def loss(self, model, graph, mask):
        import torch.nn.functional as F
        model.train()
        clean_logits = model(graph)
        structure_delta = self.structure_perturbation(model, graph, mask) if self.parts in {'both', 'structure'} else None
        adversarial_texts = self.lexical_augmentation(model, graph, mask) if self.parts in {'both', 'text'} else None
        model.train()
        adversarial_logits = model(graph, texts=adversarial_texts, x_delta=structure_delta)
        clean_loss = F.cross_entropy(clean_logits[mask], graph.y[mask])
        adversarial_loss = F.cross_entropy(adversarial_logits[mask], graph.y[mask])
        return self.text_mix_alpha * clean_loss + (1 - self.text_mix_alpha) * adversarial_loss

    def aggregation_weights(self, scores: list[Tensor]) -> Tensor:
        if self.weighted_aggregation:
            return _aggregation_weights(self, scores)
        sizes = torch.tensor(self.client_sizes or [1.0] * len(scores), dtype=torch.float)
        return sizes / sizes.sum()

    def _capped(self, mask: Tensor, cap: int) -> Tensor:
        nodes = mask.nonzero(as_tuple=False).view(-1)
        if cap <= 0 or nodes.numel() <= cap:
            return mask
        generator = torch.Generator().manual_seed(self.seed + self._draws)
        self._draws += 1
        keep = nodes[torch.randperm(nodes.numel(), generator=generator)[:cap].to(nodes.device)]
        capped = torch.zeros_like(mask)
        capped[keep] = True
        return capped

    def lexical_augmentation(self, model, graph, mask):
        return super().lexical_augmentation(model, graph, self._capped(mask, self.train_cap))

    @torch.no_grad()
    def robustness_score(self, model, graph, mask):
        return super().robustness_score(model, graph, self._capped(mask, self.eval_cap))

class CappedExperiment(FederatedExperiment):

    def __init__(self, *args: Any, target_cap: int=0, minibatch: bool=False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.target_cap = int(target_cap)
        self.minibatch = bool(minibatch)
        self.optimizer_steps = 0

    def _target_nodes(self, graph: TextAttributedGraph, client_id: int) -> list[int]:
        nodes = super()._target_nodes(graph, client_id)
        return nodes if not self.target_cap else nodes[:self.target_cap]

    def _train_client(self, graph: TextAttributedGraph, use_strum: bool):
        if not self.minibatch:
            (state, score) = super()._train_client(graph, use_strum)
            self.optimizer_steps += int(self.training_config['local_epochs'])
            return (state, score)
        import torch.nn.functional as F
        from gtae.federated import _state_to_cpu
        local_model = copy.deepcopy(self.model)
        local_model.load_state_dict(self.model.state_dict())
        local_model.train()
        optimizer = self._optimizer(local_model)
        epochs = int(self.training_config['local_epochs'])
        split = epochs // 2
        batch_size = max(1, int(self.training_config['batch_size']))
        accumulation = max(1, int(self.training_config['gradient_accumulation']))
        clip = self.training_config['gradient_clip']
        train_nodes = graph.train_mask.nonzero(as_tuple=False).view(-1)
        generator = torch.Generator().manual_seed(self.seed)
        for epoch in range(epochs):
            order = train_nodes[torch.randperm(train_nodes.numel(), generator=generator).to(train_nodes.device)]
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            for start in range(0, order.numel(), batch_size):
                batch = order[start:start + batch_size]
                mask = torch.zeros_like(graph.train_mask)
                mask[batch] = True
                if use_strum and epoch >= split:
                    loss = self.strum.loss(local_model, graph, mask)
                else:
                    logits = local_model(graph)
                    loss = F.cross_entropy(logits[mask], graph.y[mask])
                (loss / accumulation).backward()
                pending += 1
                if pending % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.optimizer_steps += 1
            if pending % accumulation:
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                self.optimizer_steps += 1
        score = self.strum.robustness_score(local_model, graph, graph.val_mask) if use_strum else torch.tensor(float(graph.train_mask.sum()), device=self.device)
        return (_state_to_cpu(local_model), score)

def _louvain_partition(graph: TextAttributedGraph, clients: int, seed: int) -> list[list[int]]:
    import networkx as nx
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx
    data = Data(edge_index=graph.edge_index.cpu(), num_nodes=graph.num_nodes)
    nx_graph = to_networkx(data, to_undirected=True)
    communities = nx.community.louvain_communities(nx_graph, seed=seed)
    return up_partition._normalize_groups([list(group) for group in communities], clients)

def apply_patches(fast: bool=True) -> None:
    up_partition.louvain_partition = _louvain_partition
    InfluenceGuidedTopologyAttack._margin = _topology_margin
    STRUM.aggregation_weights = _aggregation_weights
    if fast:
        InfluenceGuidedTopologyAttack.attack_node = _fast_topology_attack_node
