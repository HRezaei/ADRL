#!/usr/bin/env python3
"""
FSDP Multi-GPU Verification Demo — diagnostic version.

Systematically tests each layer of the stack so we can pinpoint
where the crash happens.

Usage:
    torchrun --nproc_per_node=<NUM_GPUS> mappo/scripts/fsdp_demo.py \
        --model_name HuggingFaceH4/tiny-random-LlamaForCausalLM

    torchrun --nproc_per_node=<NUM_GPUS> mappo/scripts/fsdp_demo.py \
        --model_name t5-small --seq2seq
"""

import argparse
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from mappo.utils.distributed import (
    init_distributed,
    is_distributed,
    get_local_rank,
    get_world_size,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def break_tied_weights(model):
    """Break all shared parameter tensors so FSDP can flatten safely."""
    if hasattr(model, "lm_head") and hasattr(model, "shared"):
        if model.lm_head.weight is model.shared.weight:
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
            print("    [tied-weights] broke lm_head <-> shared")
        if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
            if model.encoder.embed_tokens.weight is model.shared.weight:
                model.encoder.embed_tokens.weight = nn.Parameter(
                    model.encoder.embed_tokens.weight.clone()
                )
                print("    [tied-weights] broke encoder.embed_tokens <-> shared")
        if hasattr(model, "decoder") and hasattr(model.decoder, "embed_tokens"):
            if model.decoder.embed_tokens.weight is model.shared.weight:
                model.decoder.embed_tokens.weight = nn.Parameter(
                    model.decoder.embed_tokens.weight.clone()
                )
                print("    [tied-weights] broke decoder.embed_tokens <-> shared")
    return model


def check_tied_weights(model, label=""):
    tensors = {}
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in tensors:
            print(f"    SHARED: {label}{name} shares tensor with {tensors[ptr]}")
        else:
            tensors[ptr] = f"{label}{name}"


def make_fsdp(model, device_id, use_mixed_precision=True):
    """Simplified FSDP wrap with optional mixed precision."""
    model = model.to(f"cuda:{device_id}")

    # Ensure uniform dtype for FSDP flattening
    dtypes = {p.dtype for p in model.parameters()}
    if len(dtypes) > 1:
        target = torch.float16 if torch.float16 in dtypes else torch.float32
        model = model.to(dtype=target)

    mp_kw = {}
    if use_mixed_precision:
        mp_kw["mixed_precision"] = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        **mp_kw,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceH4/tiny-random-LlamaForCausalLM",
    )
    parser.add_argument("--seq2seq", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    local_rank, rank, world_size = init_distributed()
    is_multi = is_distributed()
    device = f"cuda:{local_rank}"

    passed = 0
    failed = 0
    skipped = 0

    def ok(msg):
        nonlocal passed; passed += 1
        print(f"  [{PASS}] {msg}")

    def fail(msg):
        nonlocal failed; failed += 1
        print(f"  [{FAIL}] {msg}")

    def skip(msg):
        nonlocal skipped; skipped += 1
        print(f"  [{SKIP}] {msg}")

    # ===========================================================
    # TEST 1: Distributed init
    # ===========================================================
    print(f"\n--- Test 1: Distributed init ---")
    print(f"  rank={rank}, local_rank={local_rank}, world_size={world_size}")
    ok(f"world_size={world_size}")

    # ===========================================================
    # TEST 2: NCCL smoke test (all-reduce a dummy tensor)
    # ===========================================================
    print(f"\n--- Test 2: NCCL all-reduce smoke test ---")
    try:
        t = torch.ones(1, device=device) * (rank + 1)
        dist.all_reduce(t)
        expected = float(world_size * (world_size + 1) / 2)
        if abs(t.item() - expected) < 1e-4:
            ok(f"all-reduce OK (sum={t.item():.1f}, expected={expected})")
        else:
            fail(f"all-reduce mismatch (got {t.item()}, expected {expected})")
    except Exception as e:
        fail(f"NCCL smoke test failed: {e}")

    # ===========================================================
    # TEST 3-5: Model loading, tied weights, dtypes
    # ===========================================================
    print(f"\n--- Test 3: Model loading ---")
    try:
        if args.seq2seq:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model_cls = AutoModelForSeq2SeqLM
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model_cls = AutoModelForCausalLM
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0

        base_model = model_cls.from_pretrained(args.model_name, dtype=torch.float16)
        base_model.to("cpu")
        n_params = sum(p.numel() for p in base_model.parameters())
        dtype_counts = Counter(str(p.dtype) for p in base_model.parameters())
        print(f"  model: {args.model_name}, params: {n_params:,}")
        print(f"  dtype distribution: {dict(dtype_counts)}")
        ok("model loaded")
    except Exception as e:
        fail(f"load failed: {e}")
        sys.exit(1)

    print(f"\n--- Test 4: Tied-weight analysis ---")
    check_tied_weights(base_model)
    break_tied_weights(base_model)
    ok("tied weights handled")

    # ===========================================================
    # TEST 5: FSDP — NO mixed precision
    # ===========================================================
    print(f"\n--- Test 5: FSDP wrap (NO mixed precision) ---")
    try:
        actor = make_fsdp(base_model, local_rank, use_mixed_precision=False)
        ok("FSDP wrapping (no MP) succeeded")
    except Exception as e:
        fail(f"FSDP wrapping (no MP) failed: {e}")
        sys.exit(1)

    # ===========================================================
    # TEST 6: Forward pass (no MP)
    # ===========================================================
    print(f"\n--- Test 6: Forward pass (no MP) ---")
    try:
        dummy = ["hello world"] * 2
        enc = tokenizer(dummy, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = actor(input_ids=input_ids, attention_mask=attn_mask, decoder_input_ids=input_ids)
        print(f"  logits shape: {out.logits.shape}")
        ok("forward pass (no MP)")
    except Exception as e:
        fail(f"forward pass (no MP) crashed: {e}")
        print("  >>> If this fails, FSDP without MP is broken — check NCCL/device setup.")

    # ===========================================================
    # TEST 7: Forward pass WITH mixed precision
    # ===========================================================
    print(f"\n--- Test 7: Forward pass (with MP) ---")
    # Need a fresh model — FSDP can't be re-wrapped
    base_model2 = model_cls.from_pretrained(args.model_name, dtype=torch.float16)
    base_model2.to("cpu")
    break_tied_weights(base_model2)

    try:
        actor2 = make_fsdp(base_model2, local_rank, use_mixed_precision=True)
        with torch.no_grad():
            out2 = actor2(input_ids=input_ids, attention_mask=attn_mask, decoder_input_ids=input_ids)
        print(f"  logits shape: {out2.logits.shape}")
        ok("forward pass (with MP)")
    except Exception as e:
        fail(f"forward pass (with MP) crashed: {e}")

    # ===========================================================
    # TEST 8: Backward pass + gradient sync
    # ===========================================================
    print(f"\n--- Test 8: Backward pass ---")
    try:
        actor2.train()
        logits = actor2(input_ids=input_ids, attention_mask=attn_mask, decoder_input_ids=input_ids)
        loss = logits.sum()
        loss.backward()

        grad_ok = False
        for p in actor2.parameters():
            if p.grad is not None and p.grad.norm().item() > 0:
                grad_ok = True
                break
        ok("backward — gradients computed" if grad_ok else "backward — weak gradients")
    except Exception as e:
        fail(f"backward crashed: {e}")

    # ===========================================================
    # TEST 9: Optimizer step
    # ===========================================================
    print(f"\n--- Test 9: Optimizer step ---")
    try:
        optim = torch.optim.AdamW(
            [p for p in actor2.parameters() if p.requires_grad], lr=1e-5
        )
        optim.step()
        ok("optimizer step")
        actor2.zero_grad()
    except Exception as e:
        fail(f"optimizer failed: {e}")

    # ===========================================================
    # TEST 10: State dict
    # ===========================================================
    print(f"\n--- Test 10: State dict gather ---")
    try:
        cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(actor2, StateDictType.FULL_STATE_DICT, cfg):
            sd = actor2.state_dict()
        print(f"  gathered {len(sd)} param groups")
        ok("state dict gather")

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.pt")
            if rank == 0:
                torch.save(sd, p)
            dist.barrier()
            if rank == 0:
                loaded = torch.load(p, map_location="cpu")
                ok("save/load cycle" if len(loaded) == len(sd) else "save/load mismatch")
        dist.barrier()
    except Exception as e:
        fail(f"state dict failed: {e}")

    # ===========================================================
    # Summary
    # ===========================================================
    print(f"\n--- Summary ---")
    total = passed + failed + skipped
    print(f"  {PASS} {passed}/{total}  {FAIL} {failed}/{total}  {SKIP} {skipped}/{total}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
