# Towards Robust Text-Attributed Federated Graph Learning: Multimodal Threats and Defense

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m nltk.downloader averaged_perceptron_tagger_eng wordnet omw-1.4 punkt
```

## Data

`src/prepare_data.py` writes `data/<dataset>-<features>/tag.pt`, a PyG `Data`
object with `x`, `edge_index`, `y`, `texts` and the train/val/test masks.

```bash
source src/env.sh
python src/prepare_data.py --datasets cora pubmed --features bow sbert
```

## Run

```bash
python src/run.py --dataset cora --processed-path data/cora-bow/tag.pt \
  --backbone meta-llama/Llama-2-7b-hf --fusion llaga --mode clean --tag cora_clean

python src/run.py --dataset cora --processed-path data/cora-bow/tag.pt \
  --backbone meta-llama/Llama-2-7b-hf --fusion llaga --mode attack --tag cora_gtae

python src/run.py --dataset cora --processed-path data/cora-bow/tag.pt \
  --backbone meta-llama/Llama-2-7b-hf --fusion llaga --mode defense --tag cora_strum
```


