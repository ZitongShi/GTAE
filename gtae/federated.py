from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .attack import GTAE
from .data import TextAttributedGraph
from .defense import STRUM
from .metrics import accuracy, attack_success_rate
from .models import GraphTextModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _state_to_cpu(model: GraphTextModel) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def aggregate_states(states: list[dict[str, torch.Tensor]], weights: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {}
    for key in states[0]:
        value = states[0][key]
        if value.is_floating_point():
            result[key] = sum(float(weight) * state[key] for weight, state in zip(weights, states))
        else:
            result[key] = value.clone()
    return result


class FederatedExperiment:
    def __init__(
        self,
        model: GraphTextModel,
        clients: list[TextAttributedGraph],
        training_config: dict[str, Any],
        attack_config: dict[str, Any],
        gtae: GTAE,
        strum: STRUM,
        device: str,
        seed: int,
    ):
        self.model = model.to(device)
        self.clients = [client.to(device) for client in clients]
        self.training_config = training_config
        self.attack_config = attack_config
        self.gtae = gtae
        self.strum = strum
        self.device = device
        self.seed = seed
        set_seed(seed)

    def _optimizer(self, model: GraphTextModel) -> torch.optim.Optimizer:
        learning_rate = self.training_config["translator_learning_rate"] if model.fusion == "graphtranslator" else self.training_config["learning_rate"]
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=self.training_config["weight_decay"],
        )

    def _train_client(self, graph: TextAttributedGraph, use_strum: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        local_model = copy.deepcopy(self.model)
        local_model.load_state_dict(self.model.state_dict())
        local_model.train()
        optimizer = self._optimizer(local_model)
        epochs = int(self.training_config["local_epochs"])
        split = epochs // 2
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            if use_strum and epoch >= split:
                loss = self.strum.loss(local_model, graph, graph.train_mask)
            else:
                logits = local_model(graph)
                loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.training_config["gradient_clip"])
            optimizer.step()
        score = self.strum.robustness_score(local_model, graph, graph.val_mask) if use_strum else torch.tensor(float(graph.train_mask.sum()), device=self.device)
        return _state_to_cpu(local_model), score

    def train(self, use_strum: bool = False) -> list[dict[str, float]]:
        history = []
        for round_index in range(int(self.training_config["rounds"])):
            states = []
            scores = []
            for graph in self.clients:
                state, score = self._train_client(graph, use_strum)
                states.append(state)
                scores.append(score)
            if use_strum:
                weights = self.strum.aggregation_weights(scores)
            else:
                sizes = torch.tensor([float(graph.train_mask.sum()) for graph in self.clients])
                weights = sizes / sizes.sum()
            self.model.load_state_dict(aggregate_states(states, weights))
            metrics = self.evaluate_clean()
            metrics["round"] = float(round_index + 1)
            metrics["mean_robustness"] = float(torch.stack([score.float().cpu() for score in scores]).mean())
            history.append(metrics)
        return history

    @torch.no_grad()
    def evaluate_clean(self) -> dict[str, float]:
        self.model.eval()
        values = []
        for graph in self.clients:
            values.append(accuracy(self.model(graph), graph.y, graph.test_mask))
        return {"clean_accuracy": float(np.mean(values)), "clean_accuracy_std": float(np.std(values))}

    def _target_nodes(self, graph: TextAttributedGraph, client_id: int) -> list[int]:
        nodes = graph.test_mask.nonzero(as_tuple=False).view(-1)
        fraction = float(self.attack_config["target_fraction"])
        count = max(1, int(len(nodes) * fraction))
        generator = torch.Generator(device=nodes.device).manual_seed(self.seed + client_id)
        order = torch.randperm(len(nodes), generator=generator, device=nodes.device)
        return nodes[order[:count]].tolist()

    def evaluate_attack(self) -> dict[str, Any]:
        self.model.eval()
        malicious = min(int(self.attack_config["malicious_clients"]), len(self.clients))
        per_client = []
        for client_id, graph in enumerate(self.clients):
            with torch.no_grad():
                clean_logits = self.model(graph)
            if client_id < malicious:
                nodes = self._target_nodes(graph, client_id)
                result = self.gtae(self.model, graph, nodes)
                with torch.no_grad():
                    attacked_logits = self.model(result.graph)
                target_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=self.device)
                target_mask[nodes] = True
                record = {
                    "client": client_id,
                    "clean_accuracy": accuracy(clean_logits, graph.y, graph.test_mask),
                    "attacked_accuracy": accuracy(attacked_logits, graph.y, graph.test_mask),
                    "attack_success_rate": attack_success_rate(clean_logits, attacked_logits, graph.y, target_mask),
                    "flipped_edges": len(result.flipped_edges),
                    "text_changes": len(result.text_changes),
                }
            else:
                record = {
                    "client": client_id,
                    "clean_accuracy": accuracy(clean_logits, graph.y, graph.test_mask),
                    "attacked_accuracy": accuracy(clean_logits, graph.y, graph.test_mask),
                    "attack_success_rate": 0.0,
                    "flipped_edges": 0,
                    "text_changes": 0,
                }
            per_client.append(record)
        return {
            "clients": per_client,
            "clean_accuracy": float(np.mean([item["clean_accuracy"] for item in per_client])),
            "attacked_accuracy": float(np.mean([item["attacked_accuracy"] for item in per_client])),
            "attack_success_rate": float(np.mean([item["attack_success_rate"] for item in per_client[:malicious]])),
        }

    def run(self, mode: str, output_dir: str | Path) -> dict[str, Any]:
        if mode == "clean":
            history = self.train(use_strum=False)
            result = {"mode": mode, "history": history, "evaluation": self.evaluate_clean()}
        elif mode == "attack":
            history = self.train(use_strum=False)
            result = {"mode": mode, "history": history, "evaluation": self.evaluate_attack()}
        elif mode == "defense":
            history = self.train(use_strum=True)
            result = {"mode": mode, "history": history, "evaluation": self.evaluate_attack()}
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "results.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        return result
