# -*- coding: utf-8 -*-
"""
src/activations.py
==================
Three pooling strategies for hidden-state extraction.

Pilot confound addressed here:
  The pilot used mean-pooling over ALL tokens, including the special
  role-marking tokens (<|im_start|>, <|im_end|>, "system", "user", etc.).
  These tokens directly encode role identity before any semantic computation,
  so a probe trained on their pooled representation measures lexical identity,
  not semantic authority representation.

  This module provides three strategies:
    1. "mean_all"       — identical to pilot (baseline; expected to be ~100%)
    2. "content_only"   — mask out role-marking tokens before pooling
    3. "last_content"   — use the last non-special, non-role token only

  Comparing probe accuracy across the three strategies isolates how much of
  the signal comes from role tokens vs. content representations.
"""

from typing import List, Tuple
import numpy as np
import torch


# Special tokens that directly encode role identity in Qwen ChatML format.
# These are excluded in the content_only and last_content strategies.
ROLE_MARKER_STRINGS = {
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    # Role-name tokens (exact strings as they appear in the tokenizer vocab)
    "system",
    "user",
    "assistant",
    "tool",
    # Scramble placeholders
    "ROLE_A", "ROLE_B", "ROLE_C", "ROLE_D", "ROLE_X",
    "[THINK_START]", "[THINK_END]",
}


def _get_role_marker_ids(tokenizer) -> set:
    """
    Return the set of token IDs that correspond to role-marking strings.
    Note: some strings may tokenize to multiple sub-tokens; we collect all
    constituent token IDs. This is an approximation — single-token markers
    (like <|im_start|>) are cleanly excluded; multi-token role strings
    (like "system" → could be one token) are excluded if they appear as
    a single token in the vocabulary.
    """
    marker_ids = set()
    for s in ROLE_MARKER_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        marker_ids.update(ids)
    # Also add the eos/pad token ID
    if tokenizer.eos_token_id is not None:
        marker_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        marker_ids.add(tokenizer.pad_token_id)
    return marker_ids


@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    layers: List[int],
    max_len: int = 128,
    device: str = "cpu",
    pooling: str = "mean_all",   # "mean_all" | "content_only" | "last_content"
    batch_size: int = 4,
) -> np.ndarray:
    """
    Extract pooled hidden states for a list of texts at specified layers.

    Parameters
    ----------
    model       : loaded AutoModelForCausalLM
    tokenizer   : loaded AutoTokenizer
    texts       : list of raw strings to process
    layers      : list of 0-indexed layer indices to extract
    max_len     : maximum tokenization length (truncation)
    device      : "cpu" or "cuda"
    pooling     : pooling strategy ("mean_all", "content_only", "last_content")
    batch_size  : number of texts to process per forward pass

    Returns
    -------
    np.ndarray of shape (N, len(layers), hidden_size)
    """
    if pooling not in ("mean_all", "content_only", "last_content"):
        raise ValueError(f"Unknown pooling: {pooling!r}")

    marker_ids = _get_role_marker_ids(tokenizer) if pooling != "mean_all" else set()

    all_features = []
    n = len(texts)

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # outputs.hidden_states is a tuple of length n_layers+1
        # (including the embedding layer at index 0)
        hs_all = outputs.hidden_states

        batch_features = []
        for layer_idx in layers:
            hs = hs_all[layer_idx + 1]  # (B, T, D); +1 for embedding offset

            if pooling == "mean_all":
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)

            elif pooling == "content_only":
                # Build a content mask: 1 for tokens that are NOT role markers
                # and NOT padding, 0 elsewhere.
                content_mask = attention_mask.clone().float()
                for b in range(input_ids.shape[0]):
                    for t in range(input_ids.shape[1]):
                        tid = input_ids[b, t].item()
                        if tid in marker_ids:
                            content_mask[b, t] = 0.0
                content_mask_unsq = content_mask.unsqueeze(-1)
                denom = content_mask_unsq.sum(dim=1).clamp(min=1.0)
                pooled = (hs * content_mask_unsq).sum(dim=1) / denom

            elif pooling == "last_content":
                # For each sequence, find the last non-marker, non-padding token.
                pooled_list = []
                for b in range(input_ids.shape[0]):
                    last_idx = 0
                    for t in range(input_ids.shape[1]):
                        tid = input_ids[b, t].item()
                        if attention_mask[b, t].item() == 1 and tid not in marker_ids:
                            last_idx = t
                    pooled_list.append(hs[b, last_idx, :])
                pooled = torch.stack(pooled_list, dim=0)

            batch_features.append(pooled.float().cpu().numpy())

        # Stack to (B, n_layers, D)
        all_features.append(np.stack(batch_features, axis=1))

        if device == "cuda":
            torch.cuda.empty_cache()

        processed = min(start + batch_size, n)
        print(f"  Processed {processed}/{n}...", end="\r", flush=True)

    print()
    return np.concatenate(all_features, axis=0)


@torch.no_grad()
def extract_single(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    max_len: int = 256,
    device: str = "cpu",
    pooling: str = "mean_all",
) -> np.ndarray:
    """
    Extract a single pooled vector for one text at one layer. Returns (D,).
    Convenience wrapper around extract_hidden_states.
    """
    result = extract_hidden_states(
        model, tokenizer, [text], [layer_idx],
        max_len=max_len, device=device, pooling=pooling, batch_size=1,
    )
    return result[0, 0, :]  # shape (D,)
