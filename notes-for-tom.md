# Notes from spiritbuun fork — CUDA optimizations

## 1. Norm correction (zero decode cost, PPL improvement)

Simple change to SET_ROWS quantization: instead of storing the original norm, store `original_norm / ||reconstruction||`. The dequantized vector's magnitude then exactly matches the original, rather than only getting the direction right.

For turbo3, the reconstruction is just the centroid vector:

```
corrected_norm = norm / ||centroid_vector||
```

For turbo4, the full reconstruction includes the QJL residual correction, so you need to defer the norm write until after sign bits are computed:

```
corrected_norm = norm / sqrt(sum_j (centroid[idx_j] + sign_j * qjl_scale_unit)^2)
```

where `qjl_scale_unit = (sqrt(pi/2) / 128) * rnorm`.

The denominator is computed entirely from values already available during quantization (codebook indices and sign bits), so there's no extra storage. At decode time nothing changes — you multiply by the stored norm as before, but now it's the corrected one.

### Results (Qwen3.5 27B Q6_K, RTX 3090, 2K context 8 chunks)

| Type   | PPL    | vs q8_0 |
|--------|--------|---------|
| q8_0   | 5.8375 | —       |
| turbo3 | 5.8323 | -0.09%  |
| turbo4 | 5.8186 | -0.32%  |

Both turbo types beat the q8_0 baseline.

## 2. Context scaling regression fix — register centroid LUT

We saw your notes on the context scaling regression (dequant cost in FA inner loop). We fixed it by moving the centroid lookup out of constant memory and into float registers.

Instead of hitting `__constant__` memory on every element in the inner loop (which serializes when multiple warps access different addresses), we precompute `centroid[i] * norm` into a small float array in registers at the start of each block:

```c
float cn[8];  // turbo3: 8 centroids, turbo4: 8 centroids
for (int c = 0; c < 8; c++)
    cn[c] = d_turbo_centroids_3bit[c] * norm;

// inner loop just indexes into registers:
float val = cn[idx];
```

This eliminates the constant memory contention entirely. The turbo/q8 speed ratio now *improves* at longer contexts instead of degrading:

### Decode speed tg64 (tok/s)

| Context | q8_0  | turbo3 | turbo4 | t3/q8 | t4/q8 |
|---------|-------|--------|--------|-------|-------|
| 4K      | 31.02 | 29.93  | 29.43  | 0.965 | 0.949 |
| 16K     | 30.77 | 29.65  | 29.41  | 0.964 | 0.956 |
| 32K     | 30.69 | 29.83  | 29.47  | 0.972 | 0.960 |

q8_0 itself slows by 1.1% from 4K→32K (more bandwidth pressure as cache grows), while turbo3 only slows 0.3%. The gap narrows because turbo's smaller KV cache puts less pressure on memory bandwidth at long contexts.
