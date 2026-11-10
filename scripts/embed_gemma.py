"""Embeds deduplicated POI texts with google/embeddinggemma-2 (BF16, prompt "Clustering"), stored as fp32.

Widths below 768 are Matryoshka truncations, re-normalised. Never FP16 (CLAUDE.md).
Runs in its own venv (sentence-transformers >= 6.1), not the training venv.

Usage: python scripts/embed_gemma.py <texts_json> <out_prefix> [--dims 768,256,128] [--batch 64]
Writes <out_prefix>-gemma<d>.npy for each width.

Other embedders (same texts, for consistency with the paper datasets), e.g. Qwen3 with no prompt:
  --model Qwen/Qwen3-Embedding-4B --prompt_name "" --tag qwen3-4b- --dims 2560 --batch 8
--limit N embeds only the first N texts (timing and consistency checks). Prints wall time and tokens per second.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

p = argparse.ArgumentParser()
p.add_argument('texts_json')
p.add_argument('out_prefix')
p.add_argument('--dims', default='768,256,128')
p.add_argument('--batch', type=int, default=64)
p.add_argument('--model', default='google/embeddinggemma-2')
p.add_argument('--prompt_name', default='Clustering', help="'' for no prompt")
p.add_argument('--tag', default='gemma')
p.add_argument('--limit', type=int, default=0)
args = p.parse_args()
dims = [int(d) for d in args.dims.split(',')]
outs = {d: f'{args.out_prefix}-{args.tag}{d}.npy' for d in dims}
assert not any(os.path.exists(o) for o in outs.values()), "outputs exist; not overwriting"

with open(args.texts_json, encoding='utf-8') as fp:
    texts = json.load(fp)
if args.limit:
    texts = texts[:args.limit]
model = SentenceTransformer(args.model, device='cuda', model_kwargs={'torch_dtype': torch.bfloat16})
assert next(model.parameters()).dtype == torch.bfloat16

prompt = model.prompts[args.prompt_name] if args.prompt_name else ''
tok = getattr(model.tokenizer, 'tokenizer', model.tokenizer)  # the model ships a processor; use its text tokenizer
lens = np.array([len(x) for x in tok([prompt + t for t in texts])['input_ids']])  # unpadded
print(f"{len(texts)} texts; tokens p50 {np.median(lens):.0f}, p99 {np.percentile(lens, 99):.0f}, max {lens.max()}; "
      f"max_seq_length {model.max_seq_length}")
assert lens.max() <= model.max_seq_length, "texts would be truncated"

t0 = time.time()
emb = model.encode(texts, prompt_name=args.prompt_name or None, batch_size=args.batch, show_progress_bar=True,
                   convert_to_numpy=True).astype(np.float32)
sec = time.time() - t0
print(f"encode wall time {sec:.1f} s; {len(texts) / sec:.2f} texts/s; {lens.sum() / sec:.0f} tokens/s; "
      f"{lens.sum()} tokens; batch {args.batch}; {args.model}")
assert emb.shape == (len(texts), max(dims)) and np.isfinite(emb).all()
for d, out in outs.items():
    e = emb[:, :d] / np.linalg.norm(emb[:, :d], axis=1, keepdims=True)
    assert np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    np.save(out, e.astype(np.float32))
    print(f"saved {out} {e.shape}")
