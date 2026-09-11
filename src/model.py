# -*- coding: utf-8 -*-
"""
src/model.py
============
Model loading with CPU/GPU fallback. Returns a consistent tuple so all
experiments use the same loading logic.
"""

import sys
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device_info() -> dict:
    """Return reproducibility-relevant environment metadata."""
    import transformers
    import sklearn
    import numpy as np
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split(" ")[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e6, 1
        )
    return info


def load_model(
    model_id: str,
    output_hidden_states: bool = True,
):
    """
    Load a HuggingFace causal LM with automatic CPU/GPU fallback.

    Returns
    -------
    model       : loaded AutoModelForCausalLM in eval mode
    tokenizer   : loaded AutoTokenizer with pad_token set
    device      : "cuda" or "cpu"
    n_layers    : model.config.num_hidden_layers
    dtype_used  : string label for the dtype actually used
    """
    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        try:
            print("  Attempting float16 on GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                output_hidden_states=output_hidden_states,
                trust_remote_code=True,
            )
            dtype_used = "float16"
            device = "cuda"
            print("  float16 GPU load succeeded.")
        except Exception as exc:
            print(f"  GPU error ({exc}). Falling back to bfloat16 CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                output_hidden_states=output_hidden_states,
                trust_remote_code=True,
            )
            dtype_used = "bfloat16_cpu"
            device = "cpu"
    else:
        print("  No CUDA. Loading on CPU (bfloat16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            output_hidden_states=output_hidden_states,
            trust_remote_code=True,
        )
        dtype_used = "bfloat16_cpu"
        device = "cpu"

    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden: {hidden_size}, dtype: {dtype_used}")
    return model, tokenizer, device, n_layers, dtype_used
