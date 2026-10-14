"""Extract paired residual states; run from the repository root with python -m."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import random
from pathlib import Path

from src.data.schema import CSTutorContrastiveDialogue


def prepare_branch(tokenizer, item, ending=None):
    """Use the official template and locate final content tokens via offsets.

    Gemma requires a user-first conversation. A fixed topic-specific opening
    request precedes the dataset's tutor-first context in both branches.
    Context stays a dynamic list; only the exported interface uses turn numbers.
    """
    messages = [{"role": "user", "content": f"Please explain {item.topic}."}]
    turns = list(item.context_turns)
    if ending is not None:
        turns.append(ending)
    positions = []
    prefixes = []
    for turn in turns:
        role = "assistant" if turn.speaker == "tutor" else "user"
        messages.append({"role": role, "content": turn.text})
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        content = turn.text.strip()
        content_start = rendered.rfind(content)
        if not content or content_start < 0:
            raise ValueError("Cannot locate turn text inside the chat template.")
        content_end = content_start + len(content)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        candidates = [
            i for i, (start, end) in enumerate(encoded["offset_mapping"])
            if start < content_end and end > content_start
        ]
        if not candidates:
            raise ValueError("No content tokens found for turn.")
        positions.append(candidates[-1])
        prefixes.append(encoded["input_ids"][:candidates[-1] + 1])
    ids = encoded["input_ids"]
    for prefix in prefixes:
        if ids[:len(prefix)] != prefix:
            raise ValueError("Template changed earlier tokens when appending a turn.")
    return ids, positions


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def atomic_tensor(torch, path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/raw/cs_tutor_dataset.json")
    parser.add_argument("--output", default="data/processed/activations_v2")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    raw = Path(args.dataset).read_bytes()
    items = [CSTutorContrastiveDialogue.model_validate(x) for x in json.loads(raw)]
    if not items:
        raise ValueError("Dataset is empty.")
    # The midterm export contract has exactly two context states.
    for index, item in enumerate(items):
        if [t.speaker for t in item.context_turns] != ["tutor", "student"]:
            raise ValueError(f"Item {index}: export requires tutor/student context.")
        if any(t.speaker != "tutor" for t in [item.final_turn_sycophantic, item.final_turn_firm]):
            raise ValueError(f"Item {index}: endings must be tutor turns.")
        if any(not t.text.strip() for t in [*item.context_turns, item.final_turn_sycophantic, item.final_turn_firm]):
            raise ValueError(f"Item {index}: empty turn text.")
    logging.info("Validated %d dialogues across %d topics", len(items), len({x.topic for x in items}))
    if args.validate_only:
        return

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. Select a GPU Colab runtime.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    selected = items[:args.limit] if args.limit else items
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    samples = out / "samples"
    samples.mkdir(exist_ok=True)
    settings = {
        "extraction_version": 2,
        "context_strategy": "single context-only forward pass",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "model": args.model, "seed": args.seed, "max_tokens": args.max_tokens,
        "dtype": "float16" if args.device == "cuda" else "float32",
        "boundary": "last content token, excluding turn-closing tokens",
        "opening_user": "Please explain {topic}.",
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformer-lens", "transformers"]},
    }
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("Output belongs to another configuration; choose a new --output directory.")
    if not manifest.exists() and any(samples.glob("*.pt")):
        raise ValueError("Sample checkpoints lack a manifest; choose a new output directory.")
    atomic_json(manifest, settings)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("A fast tokenizer with an official chat template is required.")
    logging.info("Loading %s on %s (initial download can take several minutes)", args.model, args.device)
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device, tokenizer=tokenizer,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        fold_ln=False, center_writing_weights=False,
        center_unembed=False, fold_value_biases=False,
    )
    model.eval()
    max_length = min(args.max_tokens, model.cfg.n_ctx)
    for index, item in enumerate(selected):
        checkpoint = samples / f"{index:05d}.pt"
        if checkpoint.exists():
            logging.info("Resuming: item %d already saved", index)
            continue
        branch_states = []
        branch_info = []
        context_ids, context_positions = prepare_branch(tokenizer, item)
        # A shared context is evaluated once, without either future ending.
        # Half-precision kernels can round differently for different sequence lengths.
        context_states = None  # Will be [n_layers, context_turns, d_model].
        for ending in [None, item.final_turn_sycophantic, item.final_turn_firm]:
            ids, positions = prepare_branch(tokenizer, item, ending)
            if ids[:len(context_ids)] != context_ids:
                raise ValueError(f"Item {index}: branch changed shared context tokens.")
            if ending is not None and positions[:-1] != context_positions:
                raise ValueError(f"Item {index}: branch changed shared context boundaries.")
            if len(ids) > max_length:
                raise ValueError(f"Item {index}: {len(ids)} tokens exceeds {max_length}; no truncation performed.")
            tokens = torch.tensor([ids], device=args.device)  # [1, seq_len]
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    tokens, return_type=None,
                    names_filter=lambda name: name.endswith("hook_resid_post"),
                    pos_slice=positions if ending is None else [positions[-1]],
                )
            # Cache: [1, context_turns, d_model] or [1, 1, d_model].
            states = torch.stack([
                cache[f"blocks.{layer}.hook_resid_post"][0].float().cpu()
                for layer in range(model.cfg.n_layers)
            ])  # [n_layers, cached_positions, d_model]
            if not torch.isfinite(states).all():
                raise ValueError(f"Item {index}: non-finite activations.")
            if ending is None:
                context_states = states  # [n_layers, context_turns, d_model]
            else:
                branch_states.append(states[:, -1])  # [n_layers, d_model]
            branch_info.append({"token_count": len(ids), "positions": positions,
                                "boundary_tokens": [tokenizer.decode([ids[p]]) for p in positions]})
            del cache, tokens
        syc, firm = branch_states  # Each: [n_layers, d_model]
        atomic_tensor(torch, checkpoint, {
            "context": context_states,  # [n_layers, context_turns, d_model]
            "sycophantic": syc, "firm": firm,  # [n_layers, d_model]
            "topic": item.topic, "sample_id": index, "boundaries": branch_info,
        })
        logging.info("Saved item %d/%d; boundaries=%s", index + 1, len(selected), branch_info)

    # Per-item checkpoints are authoritative; layer exports can be rebuilt on resume.
    records = [torch.load(samples / f"{i:05d}.pt", map_location="cpu", weights_only=True)
               for i in range(len(selected))]
    for layer in range(model.cfg.n_layers):
        exported = {
            "turn1_last_token": torch.stack([r["context"][layer, 0] for r in records]),
            "turn2_last_token": torch.stack([r["context"][layer, 1] for r in records]),
            "turn3_sycophantic": torch.stack([r["sycophantic"][layer] for r in records]),
            "turn3_firm": torch.stack([r["firm"][layer] for r in records]),
            # Each tensor above: [N, d_model], CPU float32.
            "topics": [r["topic"] for r in records],
            "sample_ids": [r["sample_id"] for r in records],
        }
        atomic_tensor(torch, out / f"layer_{layer}.pt", exported)
    logging.info("Exported %d layers for %d dialogues to %s", model.cfg.n_layers, len(records), out)


if __name__ == "__main__":
    main()
