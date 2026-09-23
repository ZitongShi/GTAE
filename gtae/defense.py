from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .attack import LexicalEmbeddingAttack
from .data import TextAttributedGraph
from .metrics import prediction_consistency
from .models import GraphTextModel


class STRUM:
    def __init__(
        self,
        lexical_attack: LexicalEmbeddingAttack,
        epsilon: float = 0.1,
        adversarial_steps: int = 3,
        adversarial_step_size: float = 0.03,
        text_mix_alpha: float = 0.5,
    ):
        self.lexical_attack = lexical_attack
        self.epsilon = epsilon
        self.adversarial_steps = adversarial_steps
        self.adversarial_step_size = adversarial_step_size
        self.text_mix_alpha = text_mix_alpha

    def structure_perturbation(self, model: GraphTextModel, graph: TextAttributedGraph, mask: Tensor) -> Tensor:
        delta = torch.empty_like(graph.x).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)
        for _ in range(self.adversarial_steps):
            logits = model(graph, x_delta=delta)
            loss = F.cross_entropy(logits[mask], graph.y[mask])
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            delta = delta.detach() + self.adversarial_step_size * F.normalize(gradient, p=2, dim=-1)
            norms = delta.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            delta = delta * torch.clamp(self.epsilon / norms, max=1.0)
            delta.requires_grad_(True)
        return delta.detach()

    def lexical_augmentation(self, model: GraphTextModel, graph: TextAttributedGraph, mask: Tensor) -> list[str]:
        nodes = mask.nonzero(as_tuple=False).view(-1).tolist()
        model.eval()
        attacked, _ = self.lexical_attack(model, graph, nodes)
        return attacked.texts

    def loss(self, model: GraphTextModel, graph: TextAttributedGraph, mask: Tensor) -> Tensor:
        model.train()
        clean_logits = model(graph)
        structure_delta = self.structure_perturbation(model, graph, mask)
        adversarial_texts = self.lexical_augmentation(model, graph, mask)
        model.train()
        adversarial_logits = model(graph, texts=adversarial_texts, x_delta=structure_delta)
        clean_loss = F.cross_entropy(clean_logits[mask], graph.y[mask])
        adversarial_loss = F.cross_entropy(adversarial_logits[mask], graph.y[mask])
        return self.text_mix_alpha * clean_loss + (1 - self.text_mix_alpha) * adversarial_loss

    @torch.no_grad()
    def robustness_score(self, model: GraphTextModel, graph: TextAttributedGraph, mask: Tensor) -> Tensor:
        model.eval()
        clean_logits = model(graph)
        nodes = mask.nonzero(as_tuple=False).view(-1).tolist()
        attacked, _ = self.lexical_attack(model, graph, nodes)
        attacked_logits = model(attacked)
        return prediction_consistency(clean_logits, attacked_logits, mask)


