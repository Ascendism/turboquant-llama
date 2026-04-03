# Fork changes (this repo vs upstream)

This tree is a **downstream fork** of **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** (and its bundled **ggml** submodule / vendored sources as shipped in that project). Upstream is merged periodically so the bulk of the code is identical to stock llama.cpp at the merged revision.

The sections below describe **only the intentional differences** we maintain on top of upstream, **why** they exist, and **where** they live in the source.

---

## Summary

| Goal | Files typically involved |
|------|---------------------------|
| **TurboQuant-style** KV cache types (`GGML_TYPE_TURBO2_0`, **turbo3**, **turbo4**), FWHT / QJL paths, CUDA Flash Attention vec + prefill integration | `ggml/` (types, CUDA: `fattn*.cu` / `*.cuh`, quant kernels, `template-instances/fattn-vec-*turbo*`, etc.), `src/llama-graph.cpp`, KV cache sources |
| Load certain **GGUF** layouts (e.g. rope metadata, hybrid attention) that stock loaders reject or mis-shape | `src/llama-model-loader.cpp`, `src/llama-model.cpp` |
| Run **Qwen 3.5** / **Qwen 3.5 MoE** / **Qwen3 Next** graphs where head counts, RoPE width, and optional linear-attention tensors vary **by layer** | `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen3next.cpp` |

The large TurboQuant implementation lived on **`origin/feature/turboquant-kv-cache`** ([spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)) and is **merged into `master` here** together with upstream llama.cpp. That branch also carried **`tools/server/`** and related tweaks; this repo may differ from stock llama.cpp in server/UI files for that reason.

---

## 1. GGUF loader: `rope_dimension_sections` and `get_arr<int[4]>`

**Files:** `src/llama-model-loader.cpp`, `src/llama-model.cpp`

**What changed**

- Explicit template instantiation: `llama_model_loader::get_arr<std::array<int, 4>>(...)`.
- When loading hyperparameters for affected architectures, the code **first** tries `get_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, …)` so the array is filled **without** requiring exactly four elements in the GGUF list.
- If that fails, it falls back to the previous strict `get_key_or_arr(..., 4)` path.

**Why**

Some GGUFs (including blobs produced or transformed in ways similar to **Ollama** tooling) store **three** IM-RoPE section sizes instead of four. The strict “must be length 4” path fails loading even though the remainder can be zero-padded safely (`rope_sections` is zero-filled before this logic runs).

---

## 2. Tensor creation: per-layer attention dimensions (Qwen hybrid stacks)

**File:** `src/llama-model.cpp` (`load_tensors`, for the relevant Qwen 3.5–family / hybrid blocks)

**What changed**

- For full attention layers, projection and norm tensor shapes use **per-layer** helpers, for example:
  - `hparams.n_embd_head_k(i)`, `hparams.n_head(i)`
  - `hparams.n_embd_k_gqa(i)`, `hparams.n_embd_v_gqa(i)`
- instead of a single global `n_head` / `n_embd_head_k` / GQA width for every layer.

**Why**

Hybrid models interleave **standard multi-head attention** layers with **structured / linear attention** (gated delta net / Mamba-style) blocks. Global head counts do not match every layer’s stored weights; using layer **i**’s metadata aligns tensor dimensions with the GGUF and avoids load or runtime shape errors.

---

## 3. Optional `ssm_dt` (linear / recurrent blocks)

**Files:** `src/llama-model.cpp`, `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen3next.cpp`

**What changed**

- In `load_tensors`, the `LLM_TENSOR_SSM_DT` tensor is created with `TENSOR_NOT_REQUIRED` instead of being mandatory.
- In the graph, `alpha_biased` uses `ggml_add(..., ssm_dt)` **only when** `ssm_dt` is non-null; otherwise the branch skips the add.

**Why**

Not all checkpoints ship this bias. Requiring it breaks loading; optional handling matches optional weights.

---

## 4. Model graph: layer-local RoPE and head counts

**Files:** `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen3next.cpp`

**What changed**

- Attention build paths read **`hparams.n_embd_head_v(il)`**, **`n_head(il)`**, **`n_head_kv(il)`**, **`n_rot(il)`** (and matching asserts on `n_embd_head_k(il)`), instead of only global `n_embd_head` / `n_head` / `n_rot`.
- Reshapes/views for Q/K/V, gates, and MRoPE use those **per-layer** values.

**Why**

Matches variable geometry per layer in Qwen 3.5 / Next hybrids so attention and RoPE see the same shapes as the loaded tensors.

---

## History and syncing

- **`9952377d3`**: Qwen/hybrid GGUF snapshot (small `src/` patch) before syncing upstream.
- **`96f85f69c`**: merge of **`ggml-org/llama.cpp`** `master` (as of **`43a4ee4a2`**) on top of that snapshot.
- **`origin/feature/turboquant-kv-cache`**: full TurboQuant stack (~86 commits on top of the shared merge-base with old `master`); **merged into this repo’s `master`** after the upstream sync so CUDA KV + FA paths are present again.
- Tag **`backup/pre-upstream-merge-2026-04-03`**: points at **`9952377d3`** (Qwen-only snapshot, no TurboQuant merge).

Routine maintenance: **`git fetch upstream`** then merge **`upstream/master`** (expect conflicts in CUDA FA / KV code; re-test TurboQuant configs). See **[docs/build.md](build.md)** for build options.

---

## TurboQuant (Google Research–style KV compression)

Implemented types and training references are described in the main **[README](../README.md)** (TurboQuant section). In code, look for **`GGML_TYPE_TURBO2_0`**, **`TURBO3_0`**, **`TURBO4_0`**, **`ggml_turbo_wht`**, and CUDA flash-attention paths in **`ggml/src/ggml-cuda/`**. Paper link in README: [arXiv:2501.06815](https://arxiv.org/abs/2501.06815).

---

## License

Upstream **llama.cpp** is provided under its existing license(s) as in **`LICENSE`**. This file is documentation only and does not change licensing of third-party code.
