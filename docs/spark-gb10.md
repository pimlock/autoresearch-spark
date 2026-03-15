# Running autoresearch on NVIDIA DGX Spark (GB10)

The NVIDIA GB10 (Grace Blackwell) is a sm_121a GPU. Out of the box, `train.py`
fails with two separate issues. This document describes both and how they were
fixed.

## Issue 1: flash-attn3 not supported on sm_121a

### Symptom

```
CUDA error: no kernel image is available for execution on the device
```

The precompiled `kernels-community/flash-attn3` wheel only ships a
`torch29-cxx11-cu128-aarch64-linux` variant. Its CUDA kernels do not include a
binary for sm_121a, so the kernel launch fails at runtime.

There is no sm_121a build of flash-attn3 available at the time of writing.
A thread on the flash-attention repo ([#1969](https://github.com/Dao-AILab/flash-attention/issues/1969))
notes that `torch.nn.functional.scaled_dot_product_attention` (SDPA) is
actually **~2% faster than Flash Attention on GB10**, so SDPA is the right
choice regardless.

### Fix

Removed the `kernels` import and replaced the `fa3.flash_attn_func` call in
`CausalSelfAttention.forward` with `F.scaled_dot_product_attention`.

SDPA does not natively support sliding window attention, so a causal mask is
constructed manually for the short-window (`S`) layers:

```python
w = window_size[0]
if w < T:
    rows = torch.arange(T, device=x.device).unsqueeze(1)
    cols = torch.arange(T, device=x.device).unsqueeze(0)
    mask = (cols <= rows) & (cols > rows - w)
    attn_mask = torch.zeros(T, T, device=x.device, dtype=q.dtype).masked_fill(~mask, float('-inf'))
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
else:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Note that SDPA expects `(B, heads, T, head_dim)`, so the tensors are
transposed before and after the call.

---

## Issue 2: Triton ptxas does not know sm_121a

### Symptom

```
ptxas fatal   : Value 'sm_121a' is not defined for option 'gpu-name'
NoTritonConfigsError: No valid triton configs. PTXASError: PTXAS error: Internal Triton PTX codegen error
```

The Triton wheel bundles its own `ptxas` binary (CUDA 12.8), which predates
sm_121a support. `torch.compile` uses Triton to JIT-compile kernels and hits
this error on the first forward pass.

The system has CUDA 13.0 installed at `/usr/local/cuda-13.0/bin/ptxas`, which
does support sm_121a.

### Fix

Set `TRITON_PTXAS_PATH` at the top of `train.py` (before any Triton import)
to point at the system ptxas:

```python
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")
```

`setdefault` is used so the environment variable can still be overridden from
the shell if needed.

---

## PyTorch capability warning

On startup you will see:

```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

This is a warning, not an error. PyTorch cu128 runs fine on sm_121a once the
ptxas issue above is resolved.

---

## Summary of changes

| File | Change |
|------|--------|
| `train.py` | Removed `kernels`/flash-attn3 import; replaced attention call with SDPA; added `TRITON_PTXAS_PATH` env var; added `torch.save` checkpoint at end of training |
| `generate.py` | New file — standalone inference script with model code inlined |
