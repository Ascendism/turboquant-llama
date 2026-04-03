# Fork changes (this repo vs upstream)

This tree is a **downstream fork** of **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** (and its bundled **ggml** submodule / vendored sources as shipped in that project). Upstream is merged periodically so the bulk of the code is identical to stock llama.cpp at the merged revision.

The sections below describe **only the intentional differences** we maintain on top of upstream, **why** they exist, and **where** they live in the source.

---

## Summary

| Goal | Files typically involved |
|------|---------------------------|
| Load certain **GGUF** layouts (e.g. rope metadata, hybrid attention) that stock loaders reject or mis-shape | `src/llama-model-loader.cpp`, `src/llama-model.cpp` |
| Run **Qwen 3.5** / **Qwen 3.5 MoE** / **Qwen3 Next** graphs where head counts, RoPE width, and optional linear-attention tensors vary **by layer** | `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen3next.cpp` |

There are **no** fork-specific changes under **`tools/server/`** (`llama-server`) in the commits tracked here. If you use a customized server binary, that work lives outside this repository (another clone, branch, or uncommitted tree) unless you add it here.

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

- Custom work was recorded in commit **`9952377d3`** (*turboquant: snapshot before syncing upstream llama.cpp*) before merging **upstream `master`**; the current **`master`** includes that commit **plus** the merge of upstream.
- A git tag **`backup/pre-upstream-merge-2026-04-03`** points at the pre-merge snapshot commit for easy comparison or reset.
- To see only fork-specific file diffs against the upstream parent of that snapshot, you can use:
  - `git show 9952377d3`
  - or `git diff 9952377d3^..9952377d3`

Routine maintenance: **`git fetch upstream`** then merge **`upstream/master`** (resolve conflicts, rebuild, re-test). This fork does **not** replace upstream documentation—see **[docs/build.md](build.md)** and upstream **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** for build and contribution policy.

---

## “TurboQuant” vs what is actually in this branch

**If the name implied custom quant tech:** on current **`master`**, a full-text search of sources shows **no** identifiers, kernels, CMake flags, or docs that implement something literally called TurboQuant. The only appearance of the word is **documentation** and the snapshot **commit message** for **`9952377d3`**.

**What you *do* have here:** the changes in **`9952377d3`** (now under the upstream merge) are **model loading and Qwen 3.5 / hybrid graph** fixes, not a separate quantization stack. Matrix multiply and quant types on GPU are **ordinary ggml CUDA** paths (`GGML_CUDA=ON`, MMQ / standard quants, etc.) as in **stock llama.cpp** at your merged upstream revision—not an extra “TurboQuant layer” checked into this repo.

**If TurboQuant was or will be real code** (custom kernels, new block types, server flags, etc.), it either lives **outside this clone**, on **another branch not merged to `master`**, or still needs to be **added and documented here** with concrete files and build switches so this section can point to them.

---

## License

Upstream **llama.cpp** is provided under its existing license(s) as in **`LICENSE`**. This file is documentation only and does not change licensing of third-party code.
