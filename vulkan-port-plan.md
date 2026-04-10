# TurboQuant Vulkan Port — Implementation Plan

**Branch:** `experiment/turboquant-vulkan`
**Goal:** Port TurboQuant KV-cache compression (turbo2/3/4) from the CUDA backend to the Vulkan backend, so the fork works out-of-the-box for AMD GPUs (target: RX 580 8GB, Polaris architecture, Vulkan 1.2).

**Status legend:** `[ ]` = not started, `[~]` = in progress, `[x]` = done

---

## Context / How TurboQuant Works

TurboQuant compresses the KV cache using:
1. **FWHT rotation** (`GGML_OP_TURBO_WHT`): Applies a Walsh-Hadamard transform (with random sign flips) to each 128-element KV vector, Gaussianizing the distribution.
2. **VQ quantization** on the write path: Rotated values are scalar-quantized into `block_turbo3_0` / `block_turbo4_0` / `block_turbo2_0` structs and stored in the KV cache.
3. **On-the-fly dequant** on the read path: Flash attention kernels dequantize K/V from the turbo block format during the QK dot-product inner loop — no separate dequant buffer.

The Vulkan flash attention already dispatches on `k->type` via `pipeline_flash_attn_f32_f16[k->type]` (one pipeline per type). Adding turbo = adding dequant GLSL + registering the new pipelines.

### Key files (existing CUDA implementation — read these before implementing)

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/turbo-wht.cu` | FWHT kernel — the GLSL shader is a direct translation |
| `ggml/src/ggml-cuda/turbo-sink.cu` | Attention-sink token protection (skip for initial port) |
| `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` | InnerQ channel equalization (skip for initial port) |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*.cu` | Flash attn template instances for turbo K/V combos |
| `ggml/src/ggml-turbo-quant.c` | CPU-side: block structs, centroids, quantize/dequant |
| `ggml/src/ggml-common.h` lines 280–314 | Block struct layouts for turbo2/3/4 |
| `ggml/include/ggml.h` lines 432–434 | Type enum values (42=turbo3, 43=turbo4, 44=turbo2) |

### Key files (existing Vulkan backend — read these before implementing)

| File | Purpose |
|---|---|
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Main backend: pipeline creation, op dispatch |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` | FA shader — add turbo dequant blocks here |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` | FA scalar shader entry point (include base) |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` | Dequant for standalone ops (add turbo here too) |
| `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt` | Shader registration — add every new .comp here |

### Block struct layouts (from `ggml-common.h`)

```
block_turbo3_0  (32 values = 14 bytes):
  ggml_half norm       // 2 bytes: L2 norm
  uint8_t qs[8]        // 8 bytes: lower 2 bits of 3-bit index (4 per byte)
  uint8_t signs[4]     // 4 bytes: upper 1 bit of 3-bit index (8 per byte)

block_turbo2_0  (32 values = 10 bytes):
  ggml_half norm       // 2 bytes: L2 norm
  uint8_t qs[8]        // 8 bytes: 2-bit indices (4 per byte)

block_turbo4_0  (128 values = 66 bytes):
  ggml_half norm       // 2 bytes: L2 norm
  uint8_t qs[64]       // 64 bytes: 4-bit indices (2 per byte, low nibble first)
```

### Centroids (from `ggml-turbo-quant.c` — embed in GLSL as const arrays)

```glsl
// 2-bit (turbo2 indices 0–3)
const float CENTROIDS_2BIT[4] = float[](-0.133462, -0.039994, 0.039994, 0.133462);

// 3-bit (turbo3 indices 0–7, reconstructed as: lo2bit | (sign_bit << 2))
const float CENTROIDS_3BIT[8] = float[](
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685);

// 4-bit (turbo4 indices 0–15)
const float CENTROIDS_4BIT[16] = float[](
    -0.241556, -0.182907, -0.143047, -0.111065,
    -0.083317, -0.058069, -0.034311, -0.011353,
     0.011353,  0.034311,  0.058069,  0.083317,
     0.111065,  0.143047,  0.182907,  0.241556);
```

### WHT sign arrays (from `turbo-wht.cu` — same values, embed in GLSL)

```glsl
const float turbo_wht_s1[128] = float[](
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1);
const float turbo_wht_s2[128] = float[](
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1);
```

---

## Implementation Steps

### Step 1 — `GGML_OP_TURBO_WHT` Vulkan shader `[ ]`

Create `ggml/src/ggml-vulkan/vulkan-shaders/turbo_wht.comp`:

```glsl
#version 450

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Params {
    uint n_groups;
    int  direction;  // 0 = forward, 1 = inverse
} p;

layout(binding = 0) readonly  buffer Src { float src[]; };
layout(binding = 1) writeonly buffer Dst { float dst[]; };

shared float buf[128];

// Sign arrays — must match turbo-wht.cu exactly
const float s1[128] = float[](...); // paste from turbo-wht.cu d_turbo_wht_s1
const float s2[128] = float[](...); // paste from turbo-wht.cu d_turbo_wht_s2

void main() {
    uint gid = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;
    if (gid >= p.n_groups) return;

    uint offset = gid * 128;

    // Select sign arrays by direction
    float s_first  = (p.direction == 0) ? s1[lid] : s2[lid];
    float s_second = (p.direction == 0) ? s2[lid] : s1[lid];

    // Load + apply first signs
    buf[lid] = src[offset + lid] * s_first;
    barrier();

    // Butterfly passes (7 passes for N=128)
    for (uint h = 1; h < 128; h *= 2) {
        uint j = (lid / h) * (2 * h) + (lid % h);
        if (lid < 64) {
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b;
            buf[j + h] = a - b;
        }
        barrier();
    }

    // Normalize + apply second signs
    const float inv_sqrt_128 = 0.08838834764831845;
    dst[offset + lid] = buf[lid] * inv_sqrt_128 * s_second;
}
```

**Note:** The butterfly needs all 128 threads in the barrier. Use `lid < 64` guard inside the loop as in the CUDA kernel, but **all 128 threads must still call `barrier()`** — restructure so barrier() is outside the if-guard.

---

### Step 2 — Wire `GGML_OP_TURBO_WHT` into `ggml-vulkan.cpp` `[ ]`

Locations to modify in `ggml-vulkan.cpp`:

1. **Pipeline struct** (~line 843): Add `vk_pipeline pipeline_turbo_wht;` to the device struct.

2. **Pipeline creation** (find where `pipeline_flash_attn_split_k_reduce` is created, ~line 4271): Add:
   ```cpp
   ggml_vk_create_pipeline(device, device->pipeline_turbo_wht,
       "turbo_wht", turbo_wht_len, turbo_wht_data, "main",
       2, sizeof(vk_op_turbo_wht_push_constants), {128, 1, 1}, {}, 1, true);
   ```

3. **Push constants struct** (add near other push constants structs):
   ```cpp
   struct vk_op_turbo_wht_push_constants {
       uint32_t n_groups;
       int32_t  direction;
   };
   ```

4. **Op dispatch** — find the giant switch on `node->op` (around `GGML_OP_FLASH_ATTN_EXT`). Add case:
   ```cpp
   case GGML_OP_TURBO_WHT:
       ggml_vk_op_turbo_wht(ctx, subctx, node);
       break;
   ```

5. **Support check** — find `ggml_vk_op_supports` or similar function that returns false for unsupported ops. Add `GGML_OP_TURBO_WHT` to the supported list.

6. **Implement `ggml_vk_op_turbo_wht`**:
   ```cpp
   static void ggml_vk_op_turbo_wht(ggml_backend_vk_context* ctx, vk_context& subctx, ggml_tensor* dst) {
       const ggml_tensor* src = dst->src[0];
       int direction;
       memcpy(&direction, dst->op_params, sizeof(int));
       const uint32_t n_elements = ggml_nelements(src);
       const uint32_t n_groups = n_elements / 128;
       vk_op_turbo_wht_push_constants pc{n_groups, direction};
       ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_turbo_wht,
           {{src, GGML_BACKEND_TYPE_GPU, 0}, {dst, GGML_BACKEND_TYPE_GPU, 0}},
           sizeof(pc), &pc, {n_groups, 1, 1});
   }
   ```

7. **CMakeLists.txt** for shaders: add `turbo_wht.comp` to the shader list.

8. **`ggml-vulkan-shaders.hpp` regeneration**: This file is auto-generated from compiled SPIR-V. It is regenerated at cmake configure time by `vulkan-shaders-gen`. No manual edit needed — just ensure the `.comp` is registered in `CMakeLists.txt`.

---

### Step 3 — Turbo dequant in `flash_attn_base.glsl` `[ ]`

The FA shader uses `dequantize4(ib, iqs, a_offset, binding_idx)` to decode 4 values at once from K or V. Add turbo implementations.

**Packed buffer binding approach** — look at how `DATA_A_Q4_0` is handled:
- A packed16 buffer is bound (`k_data_packed16`, `v_data_packed16`)
- The block struct is accessed by index

For turbo types, we need custom packed structs. Add to the GLSL type definitions (or use raw byte access via `uint8_t` arrays where available, or pack into uint32_t manually):

#### turbo3 dequantize4 — decodes 4 consecutive values from one block

Block: 32 values, 14 bytes. Group of 4 values selected by `iqs` (0, 4, 8, ..., 28).

```glsl
#if defined(DATA_A_TURBO3_0)
#define BLOCK_BYTE_SIZE 14  // 2 + 8 + 4

const float CENTROIDS_3BIT[8] = float[](
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685);

// Layout helpers — access raw bytes of the packed buffer
// ib = block index, iqs = element offset within block (0..28, step 4)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // norm is first 2 bytes (fp16) — access via packed16 buffer
    // qs bytes start at byte 2, signs bytes start at byte 10
    // For element i in [iqs, iqs+3]:
    //   byte_qs    = qs[i / 4],  shift = (i % 4) * 2
    //   lo2 = (qs_byte >> shift) & 0x3
    //   byte_signs = signs[i / 8], shift_s = i % 8
    //   hi1 = (signs_byte >> shift_s) & 0x1
    //   idx = lo2 | (hi1 << 2)
    //   value = norm * CENTROIDS_3BIT[idx]
    ...
}
#endif
```

**Practical approach for byte access in GLSL:** Use a `layout(binding=N) readonly buffer` of `uint` (32-bit words) and bit-shift to extract bytes. Do NOT use `uint8_t` — it requires an extension not available on all Vulkan 1.2 drivers. Pack/unpack manually.

Define a packed buffer layout per type (follow the pattern of `K_PACKED16` in the existing FA shader):

```glsl
#if defined(DATA_A_TURBO3_0)
struct block_turbo3_packed {
    uint norm_qs0;    // bytes 0-3: fp16 norm + qs[0..1]
    uint qs1_qs3;     // bytes 4-7: qs[2..5]
    uint qs4_s1;      // bytes 8-11: qs[6..7] + signs[0..1]
    uint s2_pad;      // bytes 12-13: signs[2..3] (+ 2 pad bytes to align to uint)
};
layout(binding = 1) readonly buffer K_TURBO3 { block_turbo3_packed k_turbo3[]; };
layout(binding = 2) readonly buffer V_TURBO3 { block_turbo3_packed v_turbo3[]; };
#endif
```

**WARNING on alignment:** GLSL structs in SSBOs must be aligned to 4 bytes. `block_turbo3_0` is 14 bytes — not a multiple of 4. You have two options:
- Option A: Declare the SSBO as a raw `uint[]` buffer and compute byte offsets manually. This is safer and more portable.
- Option B: Pad the struct to 16 bytes on the C++ side and update `ggml_type_size` to match.

**Recommended: Option A (raw uint[] buffer).** Calculate the word and bit offset for each field given the block index and element index.

#### turbo4 dequantize4 — decodes 4 consecutive values from one block

Block: 128 values, 66 bytes. `iqs` selects a group of 4 (0, 4, 8, ..., 124).

```glsl
#if defined(DATA_A_TURBO4_0)
#define BLOCK_BYTE_SIZE 66  // 2 + 64

const float CENTROIDS_4BIT[16] = float[](
    -0.241556, -0.182907, -0.143047, -0.111065,
    -0.083317, -0.058069, -0.034311, -0.011353,
     0.011353,  0.034311,  0.058069,  0.083317,
     0.111065,  0.143047,  0.182907,  0.241556);

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // norm: fp16 at byte 0 of block
    // qs: 4-bit packed, 2 per byte, starting at byte 2
    // For element i: byte = qs[i/2], nibble = i%2==0 ? low : high
    // value = norm * CENTROIDS_4BIT[nibble]
    ...
}
#endif
```

#### turbo2 dequantize4 (optional, lower priority)

Same structure as turbo3 but without the `signs[]` array. 2-bit index directly into `CENTROIDS_2BIT[4]`.

---

### Step 4 — Flash attention shader instances for turbo types `[ ]`

For each new type, create a `.comp` file that just `#define`s the type and includes `flash_attn_base.glsl`. Follow the exact pattern of existing instances.

**Files to create:**
```
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_f32_f16_turbo3_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_f32_f16_turbo3_0_f16acc.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_f32_f16_turbo4_0.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_f32_f16_turbo4_0_f16acc.comp
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_f32_f16_turbo2_0.comp        (optional)
ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_f32_f16_turbo2_0_f16acc.comp (optional)
```

Look at e.g. `flash_attn_f32_f16_q4_0.comp` for the exact pattern to copy. It will look roughly like:

```glsl
#define DATA_A_TURBO3_0
#include "flash_attn_base.glsl"
```

Also create `_fp32` variants if the existing types have them (check `flash_attn_f32_f16_q4_0_fp32.comp`).

**Also check:** whether `flash_attn_cm1.comp` and `flash_attn_cm2.comp` (cooperative matrix paths) need turbo variants. The RX 580 does **not** support coopmat, so this is not needed for the initial target. FA_SCALAR only.

---

### Step 5 — Register turbo pipelines in `ggml-vulkan.cpp` `[ ]`

1. **Add `CREATE_FA` calls** in the pipeline initialization block (around lines 3446–3484):
   ```cpp
   CREATE_FA(GGML_TYPE_TURBO3_0, turbo3_0, FA_SCALAR, )
   CREATE_FA(GGML_TYPE_TURBO4_0, turbo4_0, FA_SCALAR, )
   // CREATE_FA(GGML_TYPE_TURBO2_0, turbo2_0, FA_SCALAR, )  // if implementing turbo2
   ```
   Add these in both the non-fp32 and the `_fp32` suffix blocks.

2. **Type size registration** — find where `ggml_vk_get_type_size` or equivalent returns bytes-per-element for each type. Turbo types have non-power-of-2 block sizes:
   - turbo3: 14 bytes / 32 elements = 3.5 bits/element
   - turbo4: 66 bytes / 128 elements ≈ 4.125 bits/element
   - turbo2: 10 bytes / 32 elements = 2.5 bits/element

   Grep for `GGML_TYPE_Q4_0` in `ggml-vulkan.cpp` to find all the places type sizes are used and add turbo entries in the same pattern.

3. **Flash attn type whitelist** — find `ggml_vk_flash_attn` function (~line 8854). It likely has a check like:
   ```cpp
   if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q4_0 && ...) { fallback; }
   ```
   Add `GGML_TYPE_TURBO3_0`, `GGML_TYPE_TURBO4_0` to this whitelist.

4. **Packed buffer layout** — inside `ggml_vk_flash_attn`, find where it sets up the K/V descriptor bindings based on type (look for `BLOCK_BYTE_SIZE` or buffer stride calculations). Add turbo block sizes.

5. **`BLOCK_BYTE_SIZE` in the shader** — the FA shader uses `BLOCK_BYTE_SIZE` to compute strides. For turbo3 this is 14; for turbo4 this is 66. These are defined per `DATA_A_*` guard in `flash_attn_base.glsl` (see Step 3).

---

### Step 6 — Register turbo types in CMakeLists.txt `[ ]`

In `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt`, find where shader sources are listed (search for `flash_attn_f32_f16_q4_0`). Add all new `.comp` files in the same pattern:

```cmake
flash_attn_f32_f16_turbo3_0.comp
flash_attn_f32_f16_turbo3_0_f16acc.comp
flash_attn_f32_f16_turbo4_0.comp
flash_attn_f32_f16_turbo4_0_f16acc.comp
turbo_wht.comp
```

---

### Step 7 — Build and smoke test `[ ]`

Build command (no CUDA, Vulkan only):
```bash
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

First test — confirm it doesn't crash:
```bash
./build/bin/llama-cli -m /path/to/model.gguf \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    -p "Hello" -n 32
```

---

### Step 8 — Correctness validation `[ ]`

Run perplexity. Expected: numbers in the same ballpark as CUDA results.

```bash
# Baseline (q8_0, Vulkan)
./build/bin/llama-perplexity -m model.gguf --cache-type-k q8_0 --cache-type-v q8_0 \
    -f wikitext-2.txt --chunks 8 | grep "Final estimate"

# TurboQuant (Vulkan)
./build/bin/llama-perplexity -m model.gguf --cache-type-k turbo3 --cache-type-v turbo3 \
    -f wikitext-2.txt --chunks 8 | grep "Final estimate"
```

Expected turbo3 PPL: within ~0.5% of q8_0 baseline (based on CUDA results showing 5.8323 vs 5.8375).

---

### Step 9 — Decode speed benchmark `[ ]`

```bash
./build/bin/llama-bench -m model.gguf \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    -c 4096 -p 0 -n 64 | grep tg64
```

Compare to q8_0 baseline. On RX 580, turbo3 decode may be slower or faster depending on memory bandwidth vs compute tradeoff. Record results in `benchmark-results.md`.

---

### Step 10 — Commit and merge `[ ]`

When PPL is valid and decode runs cleanly:
```bash
git add ggml/src/ggml-vulkan/
git commit -m "feat(vulkan): add TurboQuant KV cache support (turbo2/3/4)"
```

Then merge into main (`master`) or open as a PR against the main branch.

---

## Known Risks / Gotchas

1. **`barrier()` in butterfly loop**: All 128 threads must reach `barrier()`. Inside `for (h = 1; h < 128; h *= 2)`, the `if (lid < 64)` work guard must NOT wrap the `barrier()` call. Keep barrier outside the if.

2. **turbo4 block size = 128**: Most existing FA quant types have block size 32. turbo4's block is 128 values. The `iqs` indexing in `dequantize4` will have a different stride. Double-check the `ib` (block index) calculation in the FA inner loop — it may assume `BLOCK_SIZE <= 32`.

3. **GLSL struct alignment**: SSBO structs must be 4-byte aligned. turbo3 is 14 bytes and turbo2 is 10 bytes — neither is a multiple of 4. Use raw `uint[]` buffer + manual byte arithmetic, not struct declarations.

4. **No `uint8_t` in GLSL without extension**: Do NOT use `uint8_t` directly. Unpack bytes from `uint` words using `(word >> (byte_idx * 8)) & 0xFF`.

5. **fp16 norm extraction**: The norm field is `ggml_half` (fp16). In GLSL, extract the uint16 from a uint32 word and convert: `unpackHalf2x16(word & 0xFFFF).x` or `unpackHalf2x16(word).x` depending on byte order.

6. **RX 580 subgroup size**: RX 580 has subgroup size 64, not 32. The FA scalar path should handle this fine since it's already parameterized by `SubGroupSize` spec constant.

7. **No coopmat on RX 580**: `FA_COOPMAT1`/`FA_COOPMAT2` paths don't need turbo support for this target. `FA_SCALAR` only.

8. **Sink tokens**: `GGML_TURBO_SINK_TOKENS` env var feature (attention-sink token protection) is CUDA-only. Skip for Vulkan port — it's an optimization on top, not required for correctness. The code will gracefully skip sink patches if `turbo_sink_get()` returns nullptr, but that function doesn't exist in the Vulkan backend — just don't port the sink code for now.

---

## Metal Reference (already implemented)

The Metal backend already has `GGML_OP_TURBO_WHT` implemented. If something is unclear in the CUDA kernel, check:
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` line ~1652 — `ggml_metal_op_turbo_wht`
- `ggml/src/ggml-metal/turbo-wht.h` — Metal shader source

The Metal flash attention for turbo K/V (if it exists) is also a reference. Check `ggml/src/ggml-metal/` for any turbo FA kernel.

---

## Not In Scope (for this branch)

- Sink token support (`GGML_TURBO_SINK_TOKENS`) on Vulkan
- InnerQ channel equalization on Vulkan
- `FA_COOPMAT1` / `FA_COOPMAT2` turbo shader instances
- Performance tuning (optimize for throughput after correctness is confirmed)
- turbo2 (optional — implement turbo3 and turbo4 first)
