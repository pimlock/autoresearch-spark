"""
Measure actual peak BF16 TFLOPS via large matrix multiplications.

This gives the empirical ceiling for your GPU, which is what MFU should
be measured against. Run it on your target hardware:

    uv run benchmark_flops.py

The script runs increasingly large matmuls, warms up, and reports the
best sustained TFLOPS observed. On a well-supported GPU the result
should approach the vendor-published spec. On GPUs with limited PyTorch
support (like GB10 with sm_121a), the result shows what you can actually
achieve in practice.
"""

import torch
import time


def benchmark_matmul(M, N, K, dtype=torch.bfloat16, device="cuda", warmup=5, iters=20):
    """Run matmul and return TFLOPS."""
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops_per_matmul = 2 * M * N * K  # multiply-accumulate = 2 FLOPs
    total_flops = flops_per_matmul * iters
    tflops = total_flops / elapsed / 1e12
    return tflops


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPU detected.")
        return

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    torch.set_float32_matmul_precision("high")

    # Test a range of matrix sizes to find peak throughput
    sizes = [1024, 2048, 4096, 8192]
    best_tflops = 0

    print(f"{'Size':>10s}  {'BF16 TFLOPS':>12s}")
    print("-" * 25)
    for size in sizes:
        tflops = benchmark_matmul(size, size, size, dtype=torch.bfloat16, device=device)
        best_tflops = max(best_tflops, tflops)
        print(f"{size:>10d}  {tflops:>12.1f}")

    print("-" * 25)
    print(f"{'Peak':>10s}  {best_tflops:>12.1f}")
    print()
    print(f"Use this value in train.py _GPU_PEAK_FLOPS:")
    print(
        f'    "{gpu_name.split()[1] if len(gpu_name.split()) > 1 else gpu_name}": {best_tflops:.1f}e12,'
    )


if __name__ == "__main__":
    main()
