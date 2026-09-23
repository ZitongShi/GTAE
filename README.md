# Towards Robust Text-Attributed Federated Graph Learning: Multimodal Threats and Defense
Code of GTAE and STRUM from *Towards Robust Text-Attributed Federated Graph Learning: Multimodal Threats and Defense*.

GTAE applies influence-guided edge flips followed by embedding-driven lexical perturbations. STRUM combines structure-side representation perturbation, lexical adversarial augmentation, and robustness-aware federated aggregation.

## Installation

```bash
pip install -e .
python -m nltk.downloader averaged_perceptron_tagger_eng wordnet omw-1.4
```

## Data

The loader supports `cora`, `pubmed`, `ogbn-arxiv`, `amz-computers`, and `amz-sports`. A processed file can be supplied for any dataset. It must contain a PyG `Data` object or a mapping with `x`, `edge_index`, `y`, `texts`, `train_mask`, `val_mask`, and `test_mask`. Texts can also be supplied as a JSON list through `texts_path`.

## Run

```bash
python scripts/run.py --config configs/default.yaml --mode clean
python scripts/run.py --config configs/default.yaml --mode attack
python scripts/run.py --config configs/default.yaml --mode defense
```

Outputs are written under `outputs/<mode>/`.
