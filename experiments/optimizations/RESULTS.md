# Megakernel Optimization Experiments

## Goal
Reduce grid.sync() calls from 8 per layer to theoretical minimum of 5 per layer.

Current: 8 syncs/layer x 28 layers = 224 syncs per decode step.

## Optimization 1: Redundant RMSNorm

**Status**: Implemented and tested

**Approach**: Have ALL 82 blocks compute RMSNorm redundantly instead of only block 0. The hidden state is only 1024 elements (2KB) - fits in L2 cache after first block reads it.

**Syncs eliminated**: 56 (2 per layer x 28 layers)

**Results**:
```
Position    Original    Optimized    Speedup
-------------------------------------------------
1           5.655ms     3.982ms      1.42x
10          5.708ms     4.006ms      1.42x
50          5.819ms     4.107ms      1.42x
100         5.936ms     4.227ms      1.40x
200         6.202ms     6.883ms      0.90x
-------------------------------------------------
Average     5.864ms     4.641ms      1.26x

Original:  170.5 tok/s
Optimized: 215.5 tok/s
Improvement: +44.9 tok/s (26.3%)
```

**Key finding**: Significant improvement at short-medium sequences, but degrades at longer sequences (position 200+). This is likely due to increased L2 cache pressure from redundant RMSNorm reads competing with KV cache reads.

**Trade-off**: Best for interactive use cases with short contexts. May hurt throughput for long-context workloads.

## Optimization 2: Head-Based Work Distribution

**Status**: Designed, not yet implemented

**Approach**: Assign blocks to attention heads (5 blocks per Q head) so QKV + attention can proceed head-local without grid.sync between QKV and attention.

**Expected syncs eliminated**: 28 (1 per layer)

**Complexity**: High - requires restructuring work distribution and handling GQA head mapping.

## Optimization 3: Fused Phases

**Status**: Not yet implemented

**Approach**: Fuse adjacent phases:
- QKV projection + QK norm + RoPE
- O projection + residual + post-attention RMSNorm

**Expected syncs eliminated**: ~56 (2 per layer)

## Summary

| Optimization | Syncs Eliminated | Speedup | Status |
|-------------|-----------------|---------|--------|
| Redundant RMSNorm | 56 | 1.26x (avg), 1.42x (short seq) | Done |
| Head-Based Distribution | 28 | TBD | Designed |
| Fused Phases | 56 | TBD | Not started |

## Running the Experiments

```bash
# Redundant RMSNorm
python experiments/optimizations/redundant_rmsnorm/benchmark.py

# Compare all (baseline)
python experiments/optimizations/compare_all.py
```

## Files

- `redundant_rmsnorm/kernel.cu` - Optimized kernel with redundant RMSNorm
- `redundant_rmsnorm/benchmark.py` - Correctness verification and benchmarking
- `compare_all.py` - Baseline comparison script
