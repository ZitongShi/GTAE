from __future__ import annotations

import math
import re
from dataclasses import dataclass

import nltk
import torch
from nltk.corpus import wordnet
from torch import Tensor
from torch.nn import functional as F

from .data import TextAttributedGraph
from .models import GraphTextModel


def _edge_pairs(edge_index: Tensor) -> set[tuple[int, int]]:
    return {(min(int(source), int(target)), max(int(source), int(target))) for source, target in edge_index.t().tolist() if source != target}


def _edge_index(pairs: set[tuple[int, int]], device: torch.device) -> Tensor:
    directed = []
    for source, target in sorted(pairs):
        directed.extend([(source, target), (target, source)])
    if not directed:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(directed, dtype=torch.long, device=device).t().contiguous()


def _toggle(pairs: set[tuple[int, int]], source: int, target: int) -> set[tuple[int, int]]:
    pair = (min(source, target), max(source, target))
    updated = set(pairs)
    if pair in updated:
        updated.remove(pair)
    else:
        updated.add(pair)
    return updated


@dataclass
class AttackResult:
    graph: TextAttributedGraph
    perturbed_nodes: list[int]
    flipped_edges: list[tuple[int, int]]
    text_changes: dict[int, str]


class InfluenceGuidedTopologyAttack:
    def __init__(self, budget_scale: float = 1.0, max_candidates: int = 64):
        self.budget_scale = budget_scale
        self.max_candidates = max_candidates

    def _candidates(self, graph: TextAttributedGraph, node: int) -> list[int]:
        neighbors = graph.edge_index[1, graph.edge_index[0] == node].unique().tolist()
        degree = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes).float()
        ranking = torch.argsort(degree, descending=True).tolist()
        values = []
        for candidate in neighbors + ranking:
            candidate = int(candidate)
            if candidate != node and candidate not in values:
                values.append(candidate)
            if len(values) >= self.max_candidates:
                break
        return values

    @torch.no_grad()
    def attack_node(self, model: GraphTextModel, graph: TextAttributedGraph, node: int) -> tuple[Tensor, list[tuple[int, int]]]:
        pairs = _edge_pairs(graph.edge_index)
        degree = sum(node in pair for pair in pairs)
        budget = max(1, int(math.ceil(degree * self.budget_scale)))
        flipped = []
        for _ in range(budget):
            best_pair = None
            best_gain = -torch.inf
            current_edge_index = _edge_index(pairs, graph.edge_index.device)
            current_margin = self._margin(model, graph, current_edge_index, node)
            for candidate in self._candidates(graph, node):
                trial_pairs = _toggle(pairs, node, candidate)
                trial_edge_index = _edge_index(trial_pairs, graph.edge_index.device)
                gain = self._margin(model, graph, trial_edge_index, node) - current_margin
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (node, candidate)
            if best_pair is None or best_gain <= 0:
                break
            pairs = _toggle(pairs, *best_pair)
            flipped.append(best_pair)
        return _edge_index(pairs, graph.edge_index.device), flipped

    @torch.no_grad()
    def __call__(self, model: GraphTextModel, graph: TextAttributedGraph, nodes: list[int]) -> tuple[TextAttributedGraph, list[tuple[int, int]]]:
        attacked = graph.clone()
        all_flips = []
        for node in nodes:
            attacked.edge_index, flips = self.attack_node(model, attacked, node)
            all_flips.extend(flips)
        return attacked, all_flips


class LexicalEmbeddingAttack:
    def __init__(
        self,
        budget_ratio: float = 0.15,
        synonym_candidates: int = 8,
        refinement_steps: int = 10,
        refinement_samples: int = 16,
        delta: float = 0.01,
        learning_rate: float = 0.05,
        l1_weight: float = 0.001,
        semantic_weight: float = 0.1,
    ):
        self.budget_ratio = budget_ratio
        self.synonym_candidates = synonym_candidates
        self.refinement_steps = refinement_steps
        self.refinement_samples = refinement_samples
        self.delta = delta
        self.learning_rate = learning_rate
        self.l1_weight = l1_weight
        self.semantic_weight = semantic_weight
        self.allowed_pos = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"}

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _detokenize(self, tokens: list[str]) -> str:
        value = " ".join(tokens)
        value = re.sub(r"\s+([,.;:!?%\)])", r"\1", value)
        value = re.sub(r"([\(])\s+", r"\1", value)
        return value

    def _wordnet_pos(self, tag: str):
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("R"):
            return wordnet.ADV
        return None

    def _synonyms(self, word: str, tag: str) -> list[str]:
        values = []
        for synset in wordnet.synsets(word, pos=self._wordnet_pos(tag)):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower() and candidate.isascii() and candidate not in values:
                    values.append(candidate)
                if len(values) >= self.synonym_candidates:
                    return values
        return values

    def _margin(self, model: GraphTextModel, graph: TextAttributedGraph, texts: list[str], node: int) -> Tensor:
        logits = model(graph, texts=texts)[node]
        label = int(graph.y[node])
        other = logits.clone()
        other[label] = -torch.inf
        return other.max() - logits[label]

    def _semantic_distance(self, original: Tensor, perturbed: Tensor) -> Tensor:
        return 1 - F.cosine_similarity(original, perturbed, dim=-1).mean()

    def _refine(self, model: GraphTextModel, original_words: list[str], adversarial_words: list[str], candidate_sets: list[list[str]]) -> list[str]:
        if not original_words:
            return adversarial_words
        original = model.token_embeddings(original_words).detach()
        adversarial = model.token_embeddings(adversarial_words).detach()
        theta = (adversarial - original).clone()
        for _ in range(self.refinement_steps):
            gradient = torch.zeros_like(theta)
            base = self._semantic_distance(original, original + theta)
            for _ in range(self.refinement_samples):
                direction = torch.randn_like(theta)
                direction = direction / direction.norm().clamp_min(1e-12)
                shifted = self._semantic_distance(original, original + theta + self.delta * direction)
                gradient = gradient + ((shifted - base) / self.delta) * direction
            gradient = gradient / self.refinement_samples
            theta = theta - self.learning_rate * (gradient + self.l1_weight * theta.sign())
        targets = original + theta
        refined = []
        for target, current, candidates in zip(targets, adversarial_words, candidate_sets):
            options = [current] + [candidate for candidate in candidates if candidate != current]
            embeddings = model.token_embeddings(options).detach()
            similarity = F.cosine_similarity(target.unsqueeze(0), embeddings, dim=-1)
            refined.append(options[int(similarity.argmax())])
        return refined

    @torch.no_grad()
    def attack_node(self, model: GraphTextModel, graph: TextAttributedGraph, node: int) -> str:
        tokens = self._tokens(graph.texts[node])
        tagged = nltk.pos_tag(tokens)
        replaceable = [(index, token, tag) for index, (token, tag) in enumerate(tagged) if tag in self.allowed_pos and len(token) > 2 and token.isalpha()]
        if not replaceable:
            return graph.texts[node]
        budget = max(1, int(math.ceil(len(replaceable) * self.budget_ratio)))
        working = list(tokens)
        selected = []
        selected_original = []
        selected_adversarial = []
        selected_candidates = []
        for _ in range(budget):
            best = None
            best_score = -torch.inf
            for index, word, tag in replaceable:
                if index in selected:
                    continue
                synonyms = self._synonyms(word, tag)
                for synonym in synonyms:
                    trial = list(working)
                    trial[index] = synonym
                    texts = list(graph.texts)
                    texts[node] = self._detokenize(trial)
                    margin = self._margin(model, graph, texts, node)
                    original_embedding = model.token_embeddings([word])
                    synonym_embedding = model.token_embeddings([synonym])
                    semantic = self._semantic_distance(original_embedding, synonym_embedding)
                    score = margin - self.semantic_weight * semantic
                    if score > best_score:
                        best_score = score
                        best = (index, word, synonym, synonyms)
            if best is None:
                break
            index, original, synonym, synonyms = best
            working[index] = synonym
            selected.append(index)
            selected_original.append(original)
            selected_adversarial.append(synonym)
            selected_candidates.append(synonyms)
        refined = self._refine(model, selected_original, selected_adversarial, selected_candidates)
        for index, value in zip(selected, refined):
            working[index] = value
        return self._detokenize(working)

    @torch.no_grad()
    def __call__(self, model: GraphTextModel, graph: TextAttributedGraph, nodes: list[int]) -> tuple[TextAttributedGraph, dict[int, str]]:
        attacked = graph.clone()
        changes = {}
        for node in nodes:
            value = self.attack_node(model, attacked, node)
            if value != attacked.texts[node]:
                attacked.texts[node] = value
                changes[node] = value
        return attacked, changes


class GTAE:
    def __init__(self, topology: InfluenceGuidedTopologyAttack, lexical: LexicalEmbeddingAttack):
        self.topology = topology
        self.lexical = lexical

    def __call__(self, model: GraphTextModel, graph: TextAttributedGraph, nodes: list[int]) -> AttackResult:
        model.eval()
        structure_graph, flipped = self.topology(model, graph, nodes)
        attacked_graph, text_changes = self.lexical(model, structure_graph, nodes)
        return AttackResult(
            graph=attacked_graph,
            perturbed_nodes=list(nodes),
            flipped_edges=flipped,
            text_changes=text_changes,
        )
