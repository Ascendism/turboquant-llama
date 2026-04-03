# Why this repository exists: TurboQuant in llama.cpp

**This project is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) whose reason to exist is shipping a full in-tree [TurboQuant](https://arxiv.org/abs/2504.19874)–style **KV cache** stack on **CUDA** (and related ggml integration): new quant types, kernels, Flash Attention paths, and graph hooks so you can run **`--cache-type-k` / `--cache-type-v`** with **turbo2 / turbo3 / turbo4** with Flash Attention where required.

The bulk of the tree still tracks upstream llama.cpp; the intentional deltas are **TurboQuant first**, then a **small set of compatibility patches** for certain GGUFs and Qwen 3.5 / hybrid models (see below).

**Deep dive (CUDA, block layouts, files touched):** [TURBOQUANT_CUDA_IMPLEMENTATION.md](../TURBOQUANT_CUDA_IMPLEMENTATION.md)  
**Usage, benchmarks, build flags:** [README](../README.md) (TurboQuant section near the top)

**Code map (quick):**

| Area | Where to look |
|------|----------------|
| Tensor types & CPU quant/dequant | `ggml/include/ggml.h`, `ggml/src/ggml-turbo-quant.c`, `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.h` |
| CUDA SET_ROWS / GET_ROWS / helpers | `ggml/src/ggml-cuda/turbo-quant-cuda.cuh`, `turbo-wht.cu`, `turbo-sink.cu`, `getrows.cu`, `set-rows.cu`, `ggml-cuda.cu` |
| Flash Attention + turbo K/V | `ggml/src/ggml-cuda/fattn*.cu`, `fattn*.cuh`, `template-instances/fattn-vec-*turbo*` |
| Graph / KV | `src/llama-graph.cpp`, `src/llama-kv-cache*.cpp`, `src/llama-context.cpp` |
| Metal (Apple) | `ggml/src/ggml-metal/*` (turbo-related headers/shaders on branch history) |

**Lineage:** Scaffold and types from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant); CUDA KV + FA work continued on **`feature/turboquant-kv-cache`** at [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda), now **merged into `master` here** together with periodic **upstream `master`** merges.

**Paper / blog:** [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

---

## `llama-server` with this fork

**There is no separate server codebase.** `llama-server` is the HTTP entry point in **`tools/server/`**; it links the same **`llama` / ggml** library as **`llama-cli`**, so TurboQuant KV types and CUDA paths apply identically. Build it from this tree (e.g. `cmake --build <build-dir> --config Release --target llama-server`).

**How to refer to it**

- In docs and support: **“llama-server built from [this repo](https://github.com/CarapaceUDE/turboquant-llama)”** or **“TurboQuant llama.cpp server”** if you need a short label. The executable name stays **`llama-server`** so scripts and process supervisors stay compatible.
- Avoid implying a different API product unless you fork the HTTP layer; REST behavior follows upstream **[tools/server/README.md](../tools/server/README.md)** with the same caveats as stock llama.cpp, plus this fork’s runtime options (**`--cache-type-k` / `--cache-type-v`** `turbo2` / `turbo3` / `turbo4`, Flash Attention flags, `TURBO_LAYER_ADAPTIVE`, etc.).

**Packaging (practical)**

- Ship **`llama-server`** together with **`llama-cli`** (optional) and **the same shared libraries** your platform loads (`ggml*.dll` / `.so`, CUDA deps). Version them with **one git commit** or **one release tag** for both binaries so KV/quant behavior cannot drift.
- Container images: base layer = this repo’s build; default command can be `llama-server` with env for turbo cache types.
- For public releases, a name like **`turboquant-llama-<version>`** for the *artifact zip* is fine; inside, keep **`llama-server`** as the binary name unless you add a tiny wrapper script.

**Developer notes:** [tools/server/README-dev.md](../tools/server/README-dev.md)

---

## Secondary patches (non–TurboQuant)

These exist so specific **GGUF** and **Qwen 3.5 / MoE / Next hybrid** checkpoints load and run correctly; they are not the primary goal of the fork.

### A. GGUF loader: `rope_dimension_sections`

**Files:** `src/llama-model-loader.cpp`, `src/llama-model.cpp`

Try `get_arr` for `LLM_KV_ROPE_DIMENSION_SECTIONS` before strict length-4 `get_key_or_arr`, so GGUFs with **three** IM-RoPE section entries still load (remainder stays zero-padded).

### B. Qwen hybrid: per-layer attention tensor shapes

**File:** `src/llama-model.cpp` (`load_tensors`)

Full attention layers use per-layer `n_embd_head_k(i)`, `n_head(i)`, GQA widths, so dimensions match hybrid stacks (attention vs recurrent blocks).

### C. Optional `ssm_dt`

**Files:** `src/llama-model.cpp`, `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen3next.cpp`

`LLM_TENSOR_SSM_DT` is optional; graph uses the bias only when present.

### D. Layer-local RoPE / heads in graph builders

**Files:** `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen3next.cpp`

Attention uses `hparams.n_embd_head_v(il)`, `n_head(il)`, `n_rot(il)`, etc., instead of global-only constants.

The TurboQuant merge may also include **`tools/server/`** diffs vs stock llama.cpp from the same branch history.

---

## History and syncing

- **`9952377d3`**: Qwen/hybrid `src/` snapshot (before a large upstream merge).
- **`96f85f69c`**: merge upstream **`ggml-org/llama.cpp`** (tip **`43a4ee4a2`** at the time).
- Merge of **`origin/feature/turboquant-kv-cache`**: restores full TurboQuant stack on top of that line (resolve carefully vs upstream BF16 FA, etc.).
- **`backup/pre-upstream-merge-2026-04-03`**: tag on **`9952377d3`** only (no TurboQuant merge).

Ongoing: **`git fetch upstream`**, merge **`upstream/master`**, rebuild, and re-test **turbo3/turbo4 + `-fa`** workloads. See **[docs/build.md](build.md)**.

---

## License

Upstream **llama.cpp** remains under its **`LICENSE`**. This file is documentation only.
