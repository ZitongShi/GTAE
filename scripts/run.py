from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gtae.attack import GTAE, InfluenceGuidedTopologyAttack, LexicalEmbeddingAttack
from gtae.config import load_config
from gtae.data import load_text_attributed_graph
from gtae.defense import STRUM
from gtae.federated import FederatedExperiment
from gtae.models import GraphTextModel
from gtae.partition import partition_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["clean", "attack", "defense"], default="clean")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = args.device or config.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    seed = int(config.get("seed", 42))
    graph = load_text_attributed_graph(config["dataset"], seed=seed)
    clients = partition_graph(
        graph,
        clients=int(config["dataset"]["clients"]),
        method=config["dataset"]["partition"],
        seed=seed,
        dirichlet_alpha=float(config["dataset"].get("dirichlet_alpha", 0.5)),
    )
    model = GraphTextModel(
        input_dim=graph.x.size(-1),
        num_classes=graph.num_classes,
        backbone=config["model"]["backbone"],
        fusion=config["model"]["fusion"],
        hidden_dim=int(config["model"]["hidden_dim"]),
        graph_layers=int(config["model"]["graph_layers"]),
        graph_heads=int(config["model"]["graph_heads"]),
        dropout=float(config["model"]["dropout"]),
        freeze_text_encoder=bool(config["model"]["freeze_text_encoder"]),
        max_length=int(config["model"]["max_length"]),
    )
    lexical = LexicalEmbeddingAttack(
        budget_ratio=float(config["attack"]["text_budget"]),
        synonym_candidates=int(config["attack"]["synonym_candidates"]),
        refinement_steps=int(config["attack"]["refinement_steps"]),
        refinement_samples=int(config["attack"]["refinement_samples"]),
        delta=float(config["attack"]["refinement_delta"]),
        learning_rate=float(config["attack"]["refinement_lr"]),
        l1_weight=float(config["attack"]["refinement_l1"]),
        semantic_weight=float(config["attack"]["semantic_weight"]),
    )
    gtae = GTAE(
        InfluenceGuidedTopologyAttack(
            budget_scale=float(config["attack"]["structure_budget_scale"]),
            max_candidates=int(config["attack"]["structure_candidates"]),
        ),
        lexical,
    )
    strum = STRUM(
        lexical_attack=lexical,
        epsilon=float(config["defense"]["epsilon"]),
        adversarial_steps=int(config["defense"]["adversarial_steps"]),
        adversarial_step_size=float(config["defense"]["adversarial_step_size"]),
        text_mix_alpha=float(config["defense"]["text_mix_alpha"]),
    )
    output_dir = args.output_dir or str(Path("outputs") / args.mode)
    experiment = FederatedExperiment(
        model=model,
        clients=clients,
        training_config=config["training"],
        attack_config=config["attack"],
        gtae=gtae,
        strum=strum,
        device=device,
        seed=seed,
    )
    result = experiment.run(args.mode, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

