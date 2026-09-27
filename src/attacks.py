from __future__ import annotations
import math
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
import runtime
from gtae.attack import InfluenceGuidedTopologyAttack
from gtae.data import TextAttributedGraph
from gtae.models import GraphTextModel

class SurrogateGCN(nn.Module):

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int=16, dropout: float=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, edge_index)

def fit_surrogate(graph: TextAttributedGraph, num_classes: int, hidden_dim: int=16, epochs: int=200, lr: float=0.01, weight_decay: float=0.0005, seed: int=42) -> tuple[SurrogateGCN, dict[str, float]]:
    torch.manual_seed(seed)
    grad = torch.enable_grad()
    device = graph.x.device
    model = SurrogateGCN(graph.x.size(-1), num_classes, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    (best_state, best_val) = (None, -1.0)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with grad:
            logits = model(graph.x, graph.edge_index)
            F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask]).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(graph.x, graph.edge_index).argmax(dim=-1)
        mask = graph.val_mask if bool(graph.val_mask.any()) else graph.train_mask
        value = float((predictions[mask] == graph.y[mask]).float().mean())
        if value > best_val:
            best_val = value
            best_state = {k: v.detach().clone() for (k, v) in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index).argmax(dim=-1)
    stats = {'surrogate_val_acc': best_val, 'surrogate_test_acc': float((predictions[graph.test_mask] == graph.y[graph.test_mask]).float().mean()) if bool(graph.test_mask.any()) else float('nan')}
    return (model, stats)

class PaperTopologyAttack(InfluenceGuidedTopologyAttack):

    def __init__(self, budget_scale: float=1.0, max_candidates: int=0, hidden_dim: int=16, epochs: int=200, seed: int=42):
        super().__init__(budget_scale=budget_scale, max_candidates=max_candidates or 1 << 30)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.surrogate: SurrogateGCN | None = None
        self.surrogate_stats: dict[str, float] = {}
        self._edge_evaluations = 0

    def fit(self, graph: TextAttributedGraph, num_classes: int) -> None:
        (self.surrogate, self.surrogate_stats) = fit_surrogate(graph, num_classes, hidden_dim=self.hidden_dim, epochs=self.epochs, seed=self.seed)

    @staticmethod
    def _undirected(pairs: Tensor) -> Tensor:
        return torch.cat([pairs, pairs.flip(0)], dim=1)

    @torch.no_grad()
    def attack_node(self, model: GraphTextModel, graph: TextAttributedGraph, node: int) -> tuple[Tensor, list[tuple[int, int]]]:
        assert self.surrogate is not None, 'call fit() before attacking'
        device = graph.edge_index.device
        edge_index = graph.edge_index
        label = int(graph.y[node])

        def logprobs(structure: Tensor) -> Tensor:
            return self.surrogate(graph.x, structure)[node].log_softmax(dim=-1)
        clean = logprobs(edge_index)
        other = clean.clone()
        other[label] = -torch.inf
        target = int(other.argmax())

        def margin(structure: Tensor) -> float:
            self._edge_evaluations += 1
            value = logprobs(structure)
            return float(value[target] - value[label])
        incident = (edge_index[0] == node) | (edge_index[1] == node)
        degree = int(incident.sum()) // 2
        budget = max(1, int(math.ceil(max(degree, 1) * self.budget_scale)))
        candidates = torch.arange(graph.num_nodes, device=device)
        candidates = candidates[candidates != node]
        if candidates.numel() > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        flipped: list[tuple[int, int]] = []
        for _ in range(budget):
            current = margin(edge_index)
            neighbours = set(edge_index[1, edge_index[0] == node].tolist())
            (best_gain, best_target, best_structure) = (0.0, None, None)
            for candidate in candidates.tolist():
                if candidate in neighbours:
                    keep = ~((edge_index[0] == node) & (edge_index[1] == candidate) | (edge_index[0] == candidate) & (edge_index[1] == node))
                    trial = edge_index[:, keep]
                else:
                    new = torch.tensor([[node], [candidate]], dtype=torch.long, device=device)
                    trial = torch.cat([edge_index, self._undirected(new)], dim=1)
                gain = margin(trial) - current
                if gain > best_gain:
                    (best_gain, best_target, best_structure) = (gain, candidate, trial)
            if best_structure is None:
                break
            edge_index = best_structure
            flipped.append((node, int(best_target)))
        return (edge_index, flipped)

    @torch.no_grad()
    def __call__(self, model: GraphTextModel, graph: TextAttributedGraph, nodes: list[int]):
        if self.surrogate is None:
            self.fit(graph, int(graph.y.max()) + 1)
        attacked = graph.clone()
        all_flips: list[tuple[int, int]] = []
        for node in nodes:
            (attacked.edge_index, flips) = self.attack_node(model, attacked, node)
            all_flips.extend(flips)
        return (attacked, all_flips)

class PaperLexicalAttack(runtime.FastLexicalEmbeddingAttack):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._context: dict[str, Any] | None = None
        self.refinement_gain: list[float] = []

    def _pooled_from_embeds(self, model: GraphTextModel, embeds: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            encoded = model.text_encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state.float()
        weights = mask.unsqueeze(-1).float()
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def _margin_from_pooled(self, model: GraphTextModel, node: int, pooled: Tensor, label: int) -> Tensor:
        graph_row = self._graph_hidden[node:node + 1].expand(pooled.size(0), -1)
        logits = model.classifier(model.fuse(graph_row, pooled))
        other = logits.clone()
        other[:, label] = -torch.inf
        return other.max(dim=-1).values - logits[:, label]

    def _objective(self, model: GraphTextModel, thetas: Tensor) -> Tensor:
        context = self._context
        base = context['base_embeds']
        mask = context['mask']
        positions = context['positions']
        original = context['original_embeds']
        batch = thetas.size(0)
        embeds = base.expand(batch, -1, -1).clone()
        embeds[:, positions, :] = (original.unsqueeze(0) + thetas).to(embeds.dtype)
        pooled = self._pooled_from_embeds(model, embeds, mask.expand(batch, -1))
        margin = self._margin_from_pooled(model, context['node'], pooled, context['label'])
        drift = 1 - F.cosine_similarity(original.unsqueeze(0).expand(batch, -1, -1), original.unsqueeze(0) + thetas, dim=-1).mean(dim=-1)
        return margin - self.semantic_weight * drift

    def _refine(self, model: GraphTextModel, original_words: list[str], adversarial_words: list[str], candidate_sets: list[list[str]]) -> list[str]:
        context = self._context
        if not original_words or context is None or (not context.get('positions').numel()):
            return adversarial_words
        original = context['original_embeds']
        adversarial = model.token_embeddings(adversarial_words).detach()[:original.size(0)]
        theta = (adversarial - original).clone()
        before = float(self._objective(model, theta.unsqueeze(0))[0])
        for _ in range(self.refinement_steps):
            base_value = self._objective(model, theta.unsqueeze(0))[0]
            directions = torch.randn(self.refinement_samples, *theta.shape, device=theta.device)
            directions = directions / directions.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
            shifted = self._objective(model, theta.unsqueeze(0) + self.delta * directions)
            gradient = (((shifted - base_value) / self.delta).view(-1, 1, 1) * directions).mean(dim=0)
            theta = theta + self.learning_rate * gradient - self.learning_rate * self.l1_weight * theta.sign()
        after = float(self._objective(model, theta.unsqueeze(0))[0])
        self.refinement_gain.append(after - before)
        targets = original + theta
        refined = []
        for (target, current, candidates) in zip(targets, adversarial_words, candidate_sets):
            options = [current] + [candidate for candidate in candidates if candidate != current]
            embeddings = model.token_embeddings(options).detach()
            similarity = F.cosine_similarity(target.unsqueeze(0), embeddings, dim=-1)
            refined.append(options[int(similarity.argmax())])
        return refined

    def _build_context(self, model: GraphTextModel, graph: TextAttributedGraph, node: int, text: str, words: list[str]) -> None:
        encoding = model.tokenizer(text, padding=False, truncation=True, max_length=model.max_length, return_tensors='pt', return_offsets_mapping=True)
        offsets = encoding.pop('offset_mapping')[0].tolist()
        input_ids = encoding['input_ids'].to(model.device)
        mask = encoding['attention_mask'].to(model.device)
        embedding_table = model.text_encoder.get_input_embeddings().weight
        base = embedding_table[input_ids].detach()
        positions = []
        cursor = 0
        for word in words:
            start = text.find(word, cursor)
            if start < 0:
                start = text.find(word)
            if start < 0:
                continue
            cursor = start + len(word)
            hit = next((index for (index, (a, b)) in enumerate(offsets) if a <= start < b or (a == start and b == start)), None)
            if hit is not None:
                positions.append(hit)
        positions_tensor = torch.tensor(sorted(set(positions)), dtype=torch.long, device=model.device)
        self._context = {'node': node, 'label': int(graph.y[node]), 'base_embeds': base, 'mask': mask, 'positions': positions_tensor, 'original_embeds': base[0, positions_tensor, :].float().clone() if positions_tensor.numel() else torch.zeros(0, base.size(-1), device=model.device)}

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
            (index, original, synonym, synonyms) = trials[runtime._first_argmax(scores)]
            working[index] = synonym
            selected.append(index)
            selected_original.append(original)
            selected_adversarial.append(synonym)
            selected_candidates.append(synonyms)
        self._build_context(model, graph, node, self._detokenize(working), selected_adversarial)
        refined = self._refine(model, selected_original, selected_adversarial, selected_candidates)
        self._context = None
        for (index, value) in zip(selected, refined):
            working[index] = value
        return self._detokenize(working)

def couple_features(model: GraphTextModel) -> None:
    from gtae.models import GraphEncoder
    layers = len(model.graph_encoder.layers)
    hidden_dim = model.graph_encoder.norms[0].normalized_shape[0]
    heads = model.graph_encoder.layers[0].heads
    model.graph_encoder = GraphEncoder(model.text_dim, hidden_dim, layers, heads, model.graph_encoder.dropout).to(model.device)

    def forward(self, graph, texts=None, edge_index=None, x_delta=None):
        values = graph.texts if texts is None else texts
        encoded = self.encode_texts(values)
        x = encoded if x_delta is None else encoded + x_delta
        structure = graph.edge_index if edge_index is None else edge_index
        return self.classifier(self.fuse(self.graph_encoder(x, structure), encoded))
    GraphTextModel.forward = forward

def note_track(surrogate: bool, candidates: str, refine: str, coupled: bool) -> None:
    pass

class PaperExperiment(runtime.CappedExperiment):

    def __init__(self, *args: Any, eval_nodes: int=0, malicious_clients: int=0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.eval_nodes = int(eval_nodes)
        self.malicious = int(malicious_clients) or len(self.clients)
        self._subset: list[Tensor] | None = None

    def eval_subset(self) -> list[Tensor]:
        if self._subset is not None:
            return self._subset
        totals = [int(client.test_mask.sum()) for client in self.clients]
        pool = sum(totals)
        generator = torch.Generator().manual_seed(self.seed)
        subsets = []
        for (index, client) in enumerate(self.clients):
            nodes = client.test_mask.nonzero(as_tuple=False).view(-1)
            if not self.eval_nodes or self.eval_nodes >= pool or nodes.numel() == 0:
                subsets.append(nodes)
                continue
            share = max(1, int(round(self.eval_nodes * totals[index] / pool)))
            order = torch.randperm(nodes.numel(), generator=generator).to(nodes.device)
            subsets.append(nodes[order[:min(share, nodes.numel())]])
        self._subset = subsets
        return subsets

    @torch.no_grad()
    def evaluate_clean(self) -> dict[str, float]:
        self.model.eval()
        correct = total = 0
        per_client = []
        for client in self.clients:
            logits = self.model(client)
            mask = client.test_mask
            hits = int((logits[mask].argmax(dim=-1) == client.y[mask]).sum())
            correct += hits
            total += int(mask.sum())
            per_client.append(hits / max(1, int(mask.sum())))
        return {'clean_accuracy': correct / max(1, total), 'clean_accuracy_macro': float(np.mean(per_client)), 'clean_accuracy_std': float(np.std(per_client)), 'test_nodes': total}

    def evaluate_attack(self) -> dict[str, Any]:
        self.model.eval()
        subsets = self.eval_subset()
        clean_correct = attacked_correct = evaluated = 0
        per_client = []
        for (index, (client, nodes)) in enumerate(zip(self.clients, subsets)):
            with torch.no_grad():
                clean_logits = self.model(client)
            (attacked_logits, flips, changes) = (clean_logits, 0, 0)
            if index < self.malicious and nodes.numel():
                result = self.gtae(self.model, client, nodes.tolist())
                with torch.no_grad():
                    attacked_logits = self.model(result.graph)
                (flips, changes) = (len(result.flipped_edges), len(result.text_changes))
            labels = client.y[nodes]
            clean_hits = int((clean_logits[nodes].argmax(dim=-1) == labels).sum())
            attacked_hits = int((attacked_logits[nodes].argmax(dim=-1) == labels).sum())
            clean_correct += clean_hits
            attacked_correct += attacked_hits
            evaluated += int(nodes.numel())
            per_client.append({'client': index, 'evaluated': int(nodes.numel()), 'clean_accuracy': clean_hits / max(1, int(nodes.numel())), 'attacked_accuracy': attacked_hits / max(1, int(nodes.numel())), 'flipped_edges': flips, 'text_changes': changes})
        clean = clean_correct / max(1, evaluated)
        attacked = attacked_correct / max(1, evaluated)
        full = self.evaluate_clean()
        return {'metric': 'micro-averaged accuracy over the evaluated test subset; ASR = clean - attacked, in points', 'clean_accuracy': clean, 'attacked_accuracy': attacked, 'attack_success_rate': clean - attacked, 'evaluated_nodes': evaluated, 'clean_accuracy_full_test': full['clean_accuracy'], 'clients': per_client}
