# TurboQuant CUDA Implementation for llama.cpp

## Summary

This document covers the implementation of CUDA support for TurboQuant (Google Research, ICLR 2026) KV cache quantization in TheTom's llama.cpp fork (`TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`). Both `turbo3` (3.25-bit) and `turbo4` (4.25-bit) types are now functional on NVIDIA GPUs.

**Paper**: arXiv:2504.19874  
**Blog**: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

## Base Fork

TheTom's fork was chosen because it has the most complete non-CUDA scaffolding:
- Type enums registered: `GGML_TYPE_TURBO3_0 = 41`, `GGML_TYPE_TURBO4_0 = 42`
- Block structures defined in `ggml/src/ggml-common.h`
- Metal shaders fully working (Apple Silicon)
- CLI flags (`--cache-type-k turbo3`, `--cache-type-v turbo4`)
- CPU quantize/dequantize reference implementations
- Graph-level WHT rotation nodes (Metal only)

What was missing: all CUDA kernel code. No SET_ROWS, no GET_ROWS, no flash attention support for either turbo type on NVIDIA GPUs.

## Block Structures

### turbo3 (`block_turbo3_0`, 14 bytes, QK=32)
```c
typedef struct {
    ggml_half  norm;       // 2 bytes: group norm (shared across 4 blocks in a 128-element group)
    uint8_t    qs[8];      // 8 bytes: 2-bit low indices, packed 4 per byte
    uint8_t    signs[4];   // 4 bytes: 1-bit high bit per element
} block_turbo3_0;          // 14 bytes per 32 elements = 3.5 bits/value
```
**Dequant formula**: `centroid[low2 | (hi1 << 2)] * norm`  
- `low2` = 2 bits from `qs` (element j: bits `(j%4)*2` of byte `j/4`)  
- `hi1` = 1 bit from `signs` (element j: bit `j%8` of byte `j/8`)  
- Combined 3-bit index selects from 8 Lloyd-Max centroids

**Important**: turbo3 uses 128-element "rotation groups" (`QK_TURBO3_GROUP=128`). Each group of 4 consecutive blocks shares the same group norm. The SET_ROWS kernel must process 128 elements at a time.

### turbo4 (`block_turbo4_0`, 68 bytes, QK=128)
```c
typedef struct {
    ggml_half  norm;                    //  2 bytes: primary norm
    ggml_half  rnorm;                   //  2 bytes: residual norm (for QJL correction)
    uint8_t    qs[QK_TURBO4 * 3 / 8];  // 48 bytes: 3-bit PolarQuant indices (true 3-bit packed)
    uint8_t    signs[QK_TURBO4 / 8];   // 16 bytes: 1-bit QJL signs
} block_turbo4_0;                       // 68 bytes per 128 elements = 4.25 bits/value
```
**Dequant formula**: `(centroid[3bit_idx] + sign * qjl_scale) * norm`  
- `qjl_scale = 1.2533141 / 128.0 * rnorm`  
- `sign = +1 if sign bit set, -1 otherwise`  
- 3-bit index is TRUE 3-bit packed (not 2+1 split like turbo3): `bit_offset = j*3`, spans byte boundaries

**Key difference from turbo3**: turbo4's 3-bit packing spans byte boundaries. Unpacking requires reading 2 bytes and shifting:
```c
int bit_offset = j * 3;
int byte_idx = bit_offset / 8;
int bit_pos  = bit_offset % 8;
uint16_t raw = (uint16_t)qs[byte_idx];
if (byte_idx + 1 < 48) raw |= (uint16_t)qs[byte_idx + 1] << 8;
uint8_t idx = (raw >> bit_pos) & 0x7;
```

### Shared Centroids (both types use the same 8 values)
```c
float centroids[8] = {
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685
};
```

## Files Modified/Created

All changes are in `ggml/src/ggml-cuda/`. Delivered as a Python patcher script (`apply_turbo_cuda_v2.py`).

### New files (7):
1. **`turbo-quant-cuda.cuh`** — SET_ROWS kernels (turbo3 custom + turbo4 via template), GET_ROWS dequantize functions, constants, helpers
2. **`template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu`**
3. **`template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu`**
4. **`template-instances/fattn-vec-instance-turbo3_0-q8_0.cu`** (mixed K/V)
5. **`template-instances/fattn-vec-instance-turbo4_0-q8_0.cu`** (mixed K/V)
6. **`template-instances/fattn-vec-instance-q8_0-turbo3_0.cu`** (mixed K/V)
7. **`template-instances/fattn-vec-instance-q8_0-turbo4_0.cu`** (mixed K/V)

### Modified files (7):
1. **`set-rows.cu`** — include + turbo3 custom kernel dispatch (128-element group norm) + turbo4 via `set_rows_cuda_quant` template
2. **`getrows.cu`** — include + turbo3/turbo4 cases in type switch
3. **`ggml-cuda.cu`** — SET_ROWS and GET_ROWS `supports_op` additions for both types
4. **`fattn-common.cuh`** — centroids constant, `vec_dot_fattn_vec_KQ_turbo3_0/turbo4_0` (with norm caching for turbo4), `dequantize_V_turbo3_0/turbo4_0`, dispatch in `get_vec_dot_KQ` and `get_dequantize_V`
5. **`fattn-vec.cuh`** — `nthreads_KQ/V`, `V_rows_per_thread`, `Q_q8_1` conditions to treat turbo like f16 (float Q path), EXTERN_DECL additions for turbo3+turbo4 as K type
6. **`fattn.cu`** — type gate, force VEC kernel for turbo (MMA/WMMA/tile don't have turbo support), vec dispatch lines for all K/V combos
7. **`CMakeLists.txt`** — glob patterns for turbo template instances

## Critical Bugs Found and Fixed

These are the bugs encountered during development, in order. Future implementers should be aware of all of them:

### 1. SET_ROWS crash: `cannot run operation SET_ROWS`
**Cause**: Missing `supports_op` entries in `ggml-cuda.cu` for both SET_ROWS and GET_ROWS.  
**Fix**: Add `GGML_TYPE_TURBO3_0` and `GGML_TYPE_TURBO4_0` to the type checks.

### 2. Linker errors: `undefined reference to ggml_cuda_flash_attn_ext_vec_case<turbo3>`
**Cause**: `CMakeLists.txt` only globs `*q4_0-q4_0.cu`, `*q8_0-q8_0.cu`, `*f16-f16.cu` in the default build. Turbo template instance files weren't compiled.  
**Fix**: Add `file(GLOB)` + `list(APPEND)` for turbo patterns.  
**Secondary bug**: Initial fix inserted the turbo glob BETWEEN the f16 `file(GLOB)` and its `list(APPEND)`, causing the turbo glob to overwrite `SRCS` before f16 got appended → f16 FA broke with undefined symbols for `ggml_type 1`. Fix: insert AFTER the f16 `list(APPEND)`.

### 3. `static_assert` failure: `ne==8`
**Cause**: FA vec kernel calls `dequantize_V` with `ne=8` when using the f16-style `V_rows_per_thread` path. The turbo3 dequant only asserted `ne == 2 || ne == 4`.  
**Fix**: Change assertion to `ne == 2 || ne == 4 || ne == 8`. The dequant logic already works for ne=8 (max offset j0+7=31, within QK_TURBO3=32).

### 4. Segfault at `rip=0x0` during prefill
**Cause**: During prefill (batch size > 1), the FA kernel selector picks the MMA kernel (faster for large batches). MMA tries to convert K/V to fp16 via `ggml_get_to_fp16_cuda(turbo3)` → returns NULL → null function pointer call → segfault.  
**Fix**: Add early return in `ggml_cuda_get_best_fattn_kernel` that forces turbo types to always use the VEC kernel before MMA/WMMA/tile can be considered:
```c
if (K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0 ||
    K->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO4_0) {
    if (can_use_vector_kernel) return BEST_FATTN_KERNEL_VEC;
    return BEST_FATTN_KERNEL_NONE;
}
```

### 5. Gibberish output: Q stride mismatch
**Cause**: The FA vec kernel's f16 path loads Q into per-thread registers in chunks of `cpy_ne` (typically 4) consecutive float2 pairs, using an outer loop `k_KQ_0 += nthreads*cpy_ne` and inner loop `k_KQ_1 = 0..cpy_ne`. The initial turbo3 vec_dot used `k_KQ_0 += nthreads` with no inner loop. Thread N was dotting K[2N] with Q[8N] instead of Q[2N]. Only thread 0 was correct.  
**Fix**: Match the f16 loop structure exactly — outer loop with `nthreads*cpy_ne` stride, inner loop over `cpy_ne`, Q indexed as `Q_v[k_KQ_0/nthreads + k_KQ_1]`.

### 6. Gibberish output: `half2` vs `float2` Q data type
**Cause**: On Turing+ GPUs, `V_DOT2_F32_F16_AVAILABLE` is defined, which means Q registers are stored as `half2` (4 bytes per pair). The turbo vec_dot read Q as `float2` (8 bytes per pair) unconditionally → read beyond bounds, scrambled values.  
**Fix**: Add `#ifdef V_DOT2_F32_F16_AVAILABLE` guard:
```c
#ifdef V_DOT2_F32_F16_AVAILABLE
    const float2 qf = __half22float2(((const half2 *) Q_v)[index]);
#else
    const float2 qf = ((const float2 *) Q_v)[index];
#endif
```

### 7. Gibberish output: FWHT rotation mismatch
**Cause**: The CUDA set_rows kernel applied forward FWHT rotation during quantization, but no inverse rotation existed on the CUDA path. On Metal, rotation is handled by a graph-level inverse WHT op that runs after KV cache reads. That graph-level op has no CUDA kernel. Result: K values stored rotated, Q values unrotated → dot product is meaningless.  
**Fix**: Removed forward rotation from CUDA quantize path. Both turbo3 and turbo4 now store unrotated coordinates. Quality is slightly lower (codebook was optimized for uniform post-rotation distribution) but output is correct. See "Remaining Work" for the proper fix.

### 8. set-rows.cu syntax error: `expected a statement`
**Cause**: The patcher's `insert_before('GGML_ABORT')` inserted the turbo3 `} else if` block INSIDE the existing `} else { GGML_ABORT }` block instead of BEFORE the `} else {` line.  
**Fix**: Find the `} else {` line that precedes `GGML_ABORT("unsupported type")` and insert before that.

## Design Decisions

### Why turbo3 SET_ROWS uses a custom kernel
Turbo3 has 32-element blocks but 128-element rotation groups. The group norm must be computed over all 128 elements, then distributed to the 4 blocks within that group. The generic `set_rows_cuda_quant` template processes one block at a time and can't compute cross-block norms. So turbo3 uses `k_set_rows_turbo3` which processes full 128-element groups.

### Why turbo4 SET_ROWS can use the generic template
Turbo4 blocks are natively 128 elements, matching the group size. No cross-block computation needed — `set_rows_cuda_quant<..., quantize_f32_turbo4_0_block>` works directly.

### Why turbo uses the float Q path (not q8_1)
The FA vec kernel has two Q loading paths: q8_1 (quantized Q, used for integer types like q4_0/q8_0) and float/f16 (used for GGML_TYPE_F16). Turbo centroids are non-linear floats that can't be meaningfully dotted using `dp4a` integer instructions. So turbo types must use the float Q path, same as f16. This is controlled by:
```c
constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_TURBO3_0 && type_K != GGML_TYPE_TURBO4_0;
```
And the corresponding `nthreads_KQ`, `nthreads_V`, `V_rows_per_thread` conditions that must also treat turbo like f16.

### Why `static` on `__constant__` arrays
The centroids and sign arrays are declared in header files included by multiple `.cu` translation units. Without `static`, they'd be duplicate symbols at link time. `static __constant__` gives each TU its own copy in constant memory (the compiler typically deduplicates these).

### Norm caching in turbo4 FA vec_dot
Turbo4 blocks are 128 elements. For D=128 models (most common), the entire attention head fits in one block. Without caching, `__half2float(norm)`, `__half2float(rnorm)`, and `qjl_scale = 1.2533141/128.0 * rnorm` are recomputed for every element pair — 64 redundant conversions. The v2 implementation tracks `prev_ib` and only reloads when the block index changes.

## Current Working Status

**Working and tested by user:**
- turbo3 K+V: coherent output ✓
- turbo4 K+V: coherent output ✓
- Mixed K/V types: implemented but not yet tested by user
- Norm caching: implemented but not yet tested by user

**Usage:**
```bash
# Apply patch
cd /root/llama-cpp-turboquant
git checkout -- .
git clean -fd ggml/src/ggml-cuda/
python3 apply_turbo_cuda_v2.py
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Test turbo3 (3.25 bits/value, 4.9x compression)
./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo3 --cache-type-v turbo3 -fa

# Test turbo4 (4.25 bits/value, 3.8x compression, QJL error correction)
./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo4 --cache-type-v turbo4 -fa

# Mixed: aggressive K quantization, safer V
./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo3 --cache-type-v q8_0 -fa

# Mixed: turbo4 K + q8_0 V
./build/bin/llama-server -m model.gguf -ngl 99 --cache-type-k turbo4 --cache-type-v q8_0 -fa
```

## Remaining Work: FWHT Rotation (The Big Quality Win)

### What it is
The TurboQuant paper's key insight is that applying a Fast Walsh-Hadamard Transform (FWHT) before quantization rotates the coordinates to a near-uniform distribution, making the Lloyd-Max codebook optimal. Without rotation, the codebook (designed for uniform data) is suboptimal for the actual Gaussian-with-outliers distribution of attention head coordinates.

### Current state
Both turbo3 and turbo4 on CUDA skip the FWHT rotation entirely. The `turbo_rotate_forward_cuda()` function exists in `turbo-quant-cuda.cuh` but is not called. Quality is "decent" but not what the paper claims.

### How Metal handles it
Metal applies forward FWHT rotation during `set_rows` (quantize), then inserts a graph-level inverse WHT op that runs after KV cache reads. TheTom's fork originally did this, then switched to a "pre-rotate-queries" approach where inverse rotation is removed from the dequant path and Q is instead rotated before the dot product.

Looking at the Metal code comments:
```metal
// turbo_rotate_inverse REMOVED — pre-rotate-queries handles this
```

### The correct CUDA approach: Pre-rotate Q, inverse-rotate output

**For Keys (Q·K^T):**
Since `dot(Π·Q, Π·K) = dot(Q, K)` for orthogonal Π (FWHT is orthogonal):
1. Re-enable `turbo_rotate_forward_cuda(x)` in `k_set_rows_turbo3` and `quantize_f32_turbo4_0_block`
2. Apply forward FWHT to Q inside the FA vec kernel before the main KV iteration loop
3. Dot product then works correctly: both Q and K are in the rotated space

**For Values (softmax(QK^T) · V):**
Since V is stored rotated: `softmax(QK^T) · Π·V = Π · softmax(QK^T) · V`
1. The FA output is the rotated result
2. Apply inverse FWHT to the FA output (per-head, 128 elements)
3. This can be a separate tiny kernel after FA, or done inline at the output writing stage

### Implementation challenges

**Q rotation in FA vec kernel:**
- Q is loaded into per-thread registers in `cpy_ne`-sized chunks, scattered across `nthreads_KQ` threads
- FWHT requires all 128 elements to be accessible simultaneously (butterfly operations mix all elements)
- Approach: Write per-thread Q registers to shared memory → `__syncthreads()` → one thread (or cooperative threads) runs FWHT in shared memory → `__syncthreads()` → threads read back rotated Q
- Must handle both `V_DOT2_F32_F16_AVAILABLE` (half2 Q) and non-V_DOT2 (float2 Q) paths
- The rotation only needs to happen ONCE per head, before the main KV iteration loop
- Cost: ~1800 FLOPs (two 128-element passes through sign arrays + one FWHT), negligible vs. attention computation

**Output inverse rotation:**
- After the FA kernel writes `dst[...]`, apply inverse FWHT to each 128-element head output
- Simplest approach: separate kernel launched after FA. Trivial to write, trivial to verify.
- Alternative: inline at the output writing stage (lines ~470-483 of fattn-vec.cuh where `dst_val` is computed), but this requires similar shared memory coordination

**Turbo4 complication:**
Turbo4 uses TWO different rotations:
1. PolarQuant rotation (signs1/signs2) — applied to normalized coordinates before 3-bit quantization
2. QJL rotation (qjl_signs1/qjl_signs2) — applied to the residual before sign extraction

The Metal code removed BOTH inverse rotations from the dequant path:
```metal
// turbo_rotate_inverse REMOVED — pre-rotate-queries handles this     (PolarQuant)
// turbo_rotate_inverse(QJL) REMOVED — pre-rotate-queries handles this (QJL)
```

For turbo4, the pre-rotate approach would need to handle both rotations correctly. The PolarQuant rotation is the same as turbo3 (pre-rotate Q, inverse-rotate output). The QJL correction residual uses a DIFFERENT rotation (different sign arrays: `turbo_qjl_wht_signs1/2`). Getting this wrong produces subtly degraded quality that looks correct but silently loses accuracy.

**Recommendation**: Implement turbo3 rotation first (single rotation, straightforward). Verify quality improvement with perplexity benchmarks. Then tackle turbo4's dual-rotation scheme.

### FWHT sign arrays
Both sign arrays are already defined in `turbo-quant-cuda.cuh` as `__constant__` arrays:
- `d_turbo_wht_signs1[128]` — first sign array for PolarQuant rotation
- `d_turbo_wht_signs2[128]` — second sign array for PolarQuant rotation

For turbo4 QJL rotation, you'll need two additional sign arrays (`turbo_qjl_wht_signs1/2`) which can be found in the Metal shader file: `ggml/src/ggml-metal/ggml-metal.metal`, search for `turbo_qjl_wht_signs`.

### FWHT algorithm (already implemented in turbo-quant-cuda.cuh)
```c
void turbo_fwht_128_cuda(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}
```
The forward and inverse transforms are identical (FWHT is its own inverse up to scaling), with the sign arrays applied before and after:
```c
// Forward rotation:
for (i) x[i] *= signs1[i];
fwht_128(x);
for (i) x[i] *= signs2[i];

// Inverse rotation (same operations, reversed order):
for (i) x[i] *= signs2[i];
fwht_128(x);
for (i) x[i] *= signs1[i];
```

## Other Community Implementations (Reference)

These were surveyed during initial research and may be useful for cross-referencing:

- **ikawrakow/ik_llama.cpp #1509**: CPU complete (18/18 tests), CUDA kernels written but unvalidated. Gist with CUDA code: https://gist.github.com/veritatisquaesitoressumus/6aa5973955007ffd858889c76aa60408
- **tonbistudio/turboquant-pytorch**: PyTorch+Triton implementation, works on RTX 4090 but different ecosystem
- **Upstream llama.cpp**: Discussion #20969, Issues #20977/#20979 tracking TurboQuant
- **mudler/llama.cpp feat/turbo-quant**: Early experimental

## Patcher Script Reference

The final working patcher is `apply_turbo_cuda_v2.py`. It:
- Creates 7 new files (1 header + 6 template instances)
- Modifies 7 existing files using line-by-line pattern matching (robust to whitespace differences between commits)
- Checks for already-applied patch state
- Auto-discovers the main CUDA backend file by content matching (works even if renamed)
- Prints OK/FAIL status for every operation

Run from repo root against a clean checkout of `feature/turboquant-kv-cache`.
