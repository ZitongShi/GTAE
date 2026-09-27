from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch_geometric.data import Data

def _mask(size: int, indices) -> torch.Tensor:
    value = torch.zeros(size, dtype=torch.bool)
    value[torch.as_tensor(indices).view(-1).long()] = True
    return value

def build(name: str, features: str, out_root: Path, source_path: Path, sbert_path: Path | None) -> Path:
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    num_nodes = int(source.num_nodes)
    texts = [str(item) for item in source.raw_texts]
    assert len(texts) == num_nodes, (len(texts), num_nodes)
    if features == 'bow':
        x = source.x.float()
    elif features == 'sbert':
        x = torch.load(sbert_path, map_location='cpu', weights_only=False).float()
        assert x.size(0) == num_nodes
    else:
        raise ValueError(features)
    if hasattr(source, 'train_mask') and source.train_mask is not None:
        (train_mask, val_mask, test_mask) = (source.train_mask, source.val_mask, source.test_mask)
    else:
        train_mask = _mask(num_nodes, source.train_id)
        val_mask = _mask(num_nodes, source.val_id)
        test_mask = _mask(num_nodes, source.test_id)
    data = Data(x=x, edge_index=source.edge_index.long(), y=source.y.view(-1).long(), train_mask=train_mask.bool(), val_mask=val_mask.bool(), test_mask=test_mask.bool())
    data.texts = texts
    out_dir = out_root / f'{name}-{features}'
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / 'tag.pt'
    torch.save(data, destination)
    stats = {'dataset': name, 'features': features, 'num_nodes': num_nodes, 'num_edges': int(data.edge_index.size(1)), 'feature_dim': int(x.size(1)), 'num_classes': int(data.y.max()) + 1, 'train': int(train_mask.sum()), 'val': int(val_mask.sum()), 'test': int(test_mask.sum()), 'source': str(source_path), 'mean_text_chars': sum((len(t) for t in texts)) / num_nodes, 'label_texts': [str(item) for item in getattr(source, 'label_texts', [])], 'example_text': texts[0][:300]}
    with (out_dir / 'stats.json').open('w', encoding='utf-8') as stream:
        json.dump(stats, stream, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return destination

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['cora', 'pubmed'])
    parser.add_argument('--features', nargs='+', default=['bow'], choices=['bow', 'sbert'])
    parser.add_argument('--out-root', default='data')
    parser.add_argument('--source', type=json.loads, required=True)
    parser.add_argument('--sbert', type=json.loads, default='{}')
    args = parser.parse_args()
    for name in args.datasets:
        for features in args.features:
            print(build(name, features, Path(args.out_root), Path(args.source[name]), Path(args.sbert[name]) if args.sbert.get(name) else None))
if __name__ == '__main__':
    main()
