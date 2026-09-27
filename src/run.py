from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
for (_i, _t) in enumerate(sys.argv):
    if _t == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import runtime
import graphllm
import attacks
from gtae.config import load_config
from gtae.data import load_text_attributed_graph
from gtae.federated import set_seed
from gtae.partition import partition_graph
DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'configs' / 'default.yaml'))
    p.add_argument('--mode', choices=['clean', 'attack', 'defense'], default='clean')
    p.add_argument('--stages', choices=['both', 'structure', 'text'], default='both')
    p.add_argument('--dataset', required=True)
    p.add_argument('--processed-path', required=True)
    p.add_argument('--backbone', default='meta-llama/Llama-2-7b-hf')
    p.add_argument('--fusion', default='llaga', choices=['llaga', 'graphprompter', 'graphtranslator'])
    p.add_argument('--clients', type=int, default=5)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--local-epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--train-nodes', type=int, default=64, help='train nodes sampled per client per epoch')
    p.add_argument('--eval-nodes', type=int, default=100)
    p.add_argument('--neighbours', type=int, default=10)
    p.add_argument('--phi', default='sentence-transformers/all-MiniLM-L6-v2', help='the encoder behind x_i = phi(t_i)')
    p.add_argument('--synonyms', default='embedding', choices=['embedding', 'wordnet'])
    p.add_argument('--synonym-threshold', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target-batch', type=int, default=8)
    p.add_argument('--encode-batch', type=int, default=16)
    p.add_argument('--strum-train-cap', type=int, default=4)
    p.add_argument('--strum-eval-cap', type=int, default=8)
    p.add_argument('--text-dtype', default='bf16', choices=sorted(DTYPES))
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--gpu', default='0')
    p.add_argument('--tag', required=True)
    return p.parse_args()

def trainable_state(model):
    return {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items() if not k.startswith('lm.')}

def clone_model(model):
    memo = {id(model.lm): model.lm, id(model.tokenizer): model.tokenizer}
    clone = model.__class__.__new__(model.__class__)
    memo[id(model)] = clone
    for (key, value) in model.__dict__.items():
        clone.__dict__[key] = copy.deepcopy(value, memo)
    return clone

@torch.no_grad()
def accuracy(model, graph, nodes, texts=None, edge_index=None, batch=8):
    if not len(nodes):
        return (0, 0)
    logits = model(graph, texts=texts, edge_index=edge_index, nodes=nodes, batch_size=batch)
    hits = int((logits.argmax(dim=-1) == graph.y[torch.as_tensor(nodes, device=graph.y.device)]).sum())
    return (hits, len(nodes))

def main():
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed
    set_seed(seed)
    runtime.apply_patches(fast=True)
    started = time.time()
    graph = load_text_attributed_graph({'name': args.dataset, 'processed_path': args.processed_path, 'root': 'data'}, seed=seed)
    stats = json.loads((Path(args.processed_path).parent / 'stats.json').read_text())
    clients = partition_graph(graph, clients=args.clients, method='louvain', seed=seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = graphllm.GraphPromptModel(num_classes=graph.num_classes, phi_name=args.phi, backbone=args.backbone, fusion=args.fusion, hidden_dim=int(config['model']['hidden_dim']), graph_layers=int(config['model']['graph_layers']), graph_heads=int(config['model']['graph_heads']), dropout=float(config['model']['dropout']), max_length=args.max_length, label_texts=stats.get('label_texts'), neighbours=args.neighbours, text_dtype=DTYPES[args.text_dtype]).to(device)
    clients = [client.to(device) for client in clients]
    loaded = time.time()
    synonyms = None
    if args.synonyms == 'embedding':
        from lexicon import EmbeddingSynonyms
        synonyms = EmbeddingSynonyms(top_k=int(config['attack']['synonym_candidates']), threshold=args.synonym_threshold, device=device)
    lexical = graphllm.FlowLexicalAttack(synonym_source=synonyms, budget_ratio=float(config['attack']['text_budget']), synonym_candidates=int(config['attack']['synonym_candidates']), refinement_steps=int(config['attack']['refinement_steps']), refinement_samples=int(config['attack']['refinement_samples']), delta=float(config['attack']['refinement_delta']), learning_rate=float(config['attack']['refinement_lr']), l1_weight=float(config['attack']['refinement_l1']), semantic_weight=float(config['attack']['semantic_weight']), encode_batch=args.encode_batch)
    topology = attacks.PaperTopologyAttack(budget_scale=float(config['attack']['structure_budget_scale']), max_candidates=0, seed=seed)
    attack = runtime.StagedGTAE(topology, lexical, stages=args.stages)
    generator = torch.Generator().manual_seed(seed)
    steps = 0
    for _round in range(args.rounds):
        (states, scores) = ([], [])
        for client in clients:
            local = clone_model(model)
            local.train()
            optimizer = torch.optim.AdamW([p for p in local.parameters() if p.requires_grad], lr=args.lr, weight_decay=float(config['training']['weight_decay']))
            train_nodes = client.train_mask.nonzero(as_tuple=False).view(-1)
            split = args.local_epochs // 2
            for epoch in range(args.local_epochs):
                order = train_nodes[torch.randperm(train_nodes.numel(), generator=generator).to(train_nodes.device)]
                batch = order[:args.train_nodes].tolist()
                texts = client.texts
                x_delta = None
                if args.mode == 'defense' and epoch >= split:
                    x_delta = structure_perturbation(local, client, batch, config, args)
                    victim = clone_model(local).eval()
                    (attacked, _) = lexical(victim, client, batch[:args.strum_train_cap])
                    texts = attacked.texts
                optimizer.zero_grad(set_to_none=True)
                clean_logits = local(client, nodes=batch, batch_size=args.target_batch)
                labels = client.y[torch.as_tensor(batch, device=client.y.device)]
                loss = F.cross_entropy(clean_logits, labels)
                if x_delta is not None:
                    adv = local(client, texts=texts, x_delta=x_delta, nodes=batch, batch_size=args.target_batch)
                    alpha = float(config['defense']['text_mix_alpha'])
                    loss = alpha * loss + (1 - alpha) * F.cross_entropy(adv, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(), float(config['training']['gradient_clip']))
                optimizer.step()
                steps += 1
            states.append(trainable_state(local))
            scores.append(robustness_score(local, client, lexical, config, args) if args.mode == 'defense' else float(train_nodes.numel()))
            del local
            torch.cuda.empty_cache()
        weights = np.array(scores, dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(scores), 1 / len(scores))
        merged = {k: sum((float(w) * s[k] for (w, s) in zip(weights, states))) for k in states[0]}
        model.load_state_dict(merged, strict=False)
    trained = time.time()
    model.eval()
    subsets = eval_subset(clients, args.eval_nodes, seed)
    clean_hits = attacked_hits = total = 0
    per_client = []
    for (index, (client, nodes)) in enumerate(zip(clients, subsets)):
        (c_hits, n) = accuracy(model, client, nodes, batch=args.target_batch)
        (a_hits, flips, changes) = (c_hits, 0, 0)
        if args.mode in {'attack', 'defense'} and n:
            result = attack(model, client, nodes)
            (a_hits, _) = accuracy(model, result.graph, nodes, texts=result.graph.texts, edge_index=result.graph.edge_index, batch=args.target_batch)
            (flips, changes) = (len(result.flipped_edges), len(result.text_changes))
        clean_hits += c_hits
        attacked_hits += a_hits
        total += n
        per_client.append({'client': index, 'evaluated': n, 'clean_accuracy': c_hits / max(1, n), 'attacked_accuracy': a_hits / max(1, n), 'flipped_edges': flips, 'text_changes': changes})
    clean = clean_hits / max(1, total)
    attacked = attacked_hits / max(1, total)
    record = {'tag': args.tag, 'args': vars(args), 'optimizer_steps_per_client': steps // max(1, len(clients)), 'surrogate': getattr(topology, 'surrogate_stats', {}), 'lm_forward_stats': {'lexical_nodes_attacked': lexical.calls, 'lexical_candidate_evaluations': lexical.candidate_evaluations}, 'paper_refinement_gain': lexical.refinement_gain, 'timings_s': {'load': loaded - started, 'train': trained - loaded, 'evaluate': time.time() - trained}, 'result': {'mode': args.mode, 'evaluation': {'metric': 'micro-averaged accuracy over the evaluated test subset; ASR = clean - attacked, in points', 'clean_accuracy': clean, 'attacked_accuracy': attacked, 'attack_success_rate': clean - attacked, 'evaluated_nodes': total, 'clients': per_client}}}
    out = Path('outputs') / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for (k, v) in record.items() if k != 'args'}, ensure_ascii=False, indent=2))

def structure_perturbation(model, client, batch, config, args):
    epsilon = float(config['defense']['epsilon'])
    delta = torch.empty_like(client.x).uniform_(-epsilon, epsilon).requires_grad_(True)
    labels = client.y[torch.as_tensor(batch, device=client.y.device)]
    for _ in range(int(config['defense']['adversarial_steps'])):
        logits = model(client, x_delta=delta, nodes=batch, batch_size=args.target_batch)
        gradient = torch.autograd.grad(F.cross_entropy(logits, labels), delta, only_inputs=True)[0]
        delta = delta.detach() + float(config['defense']['adversarial_step_size']) * F.normalize(gradient, p=2, dim=-1)
        norms = delta.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        delta = (delta * torch.clamp(epsilon / norms, max=1.0)).requires_grad_(True)
    return delta.detach()

def robustness_score(model, client, lexical, config, args) -> float:
    model.eval()
    nodes = client.val_mask.nonzero(as_tuple=False).view(-1)[:args.strum_eval_cap].tolist()
    if not nodes:
        return 1.0
    x_delta = structure_perturbation(model, client, nodes, config, args)
    with torch.no_grad():
        clean = model(client, nodes=nodes, batch_size=args.target_batch).softmax(dim=-1)
        (attacked_graph, _) = lexical(model, client, nodes)
        attacked = model(attacked_graph, texts=attacked_graph.texts, x_delta=x_delta, nodes=nodes, batch_size=args.target_batch).softmax(dim=-1)
    return float(F.cosine_similarity(clean, attacked, dim=-1).mean())

def eval_subset(clients, budget, seed):
    totals = [int(c.test_mask.sum()) for c in clients]
    pool = sum(totals)
    generator = torch.Generator().manual_seed(seed)
    out = []
    for (index, client) in enumerate(clients):
        nodes = client.test_mask.nonzero(as_tuple=False).view(-1)
        if not budget or budget >= pool or nodes.numel() == 0:
            out.append(nodes.tolist())
            continue
        share = max(1, int(round(budget * totals[index] / pool)))
        order = torch.randperm(nodes.numel(), generator=generator).to(nodes.device)
        out.append(nodes[order[:min(share, nodes.numel())]].tolist())
    return out
if __name__ == '__main__':
    main()
