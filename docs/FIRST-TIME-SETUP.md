# First-time setup (this fork)

Use this checklist to go from **zero** to a **working TurboQuant CUDA build** and optional tooling. Upstream generic build details live in **[build.md](build.md)**; this page is **fork-specific** (TurboQuant, Python pins, Web UI).

---

## 1. Clone

Pick **one** remote (same tree, different hosting):

| Remote | URL |
|--------|-----|
| **Carapace** (publishing) | `https://github.com/CarapaceUDE/turboquant-llama.git` |
| **Ascendism** | `https://github.com/Ascendism/turboquant-llama.git` |
| **spiritbuun** (lineage / backup) | `https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git` |

```bash
git clone https://github.com/CarapaceUDE/turboquant-llama.git
cd turboquant-llama
```

To merge upstream **ggml-org/llama.cpp** later, add **`upstream`** and follow **[MAINTAINING-FORK.md](MAINTAINING-FORK.md)**.

---

## 2. Prerequisites (core C++ / CUDA)

- **CMake** 3.14+ (3.20+ recommended)
- A **C++17** compiler (MSVC 2022 on Windows; GCC/Clang on Linux)
- **CUDA toolkit** matching your GPU driver (for TurboQuant + Flash Attention paths)
- **Git** (and on Windows, **Git for Windows** gives Bash if you need shell scripts)

Nothing else is required **only to compile** `llama-cli` / `llama-server`.

---

## 3. Configure and build (TurboQuant + CUDA + FA)

Use a single out-of-tree build directory (e.g. **`build/`** at the repo root).

### Linux / macOS (bash)

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

### Windows (PowerShell, Visual Studio environment)

```powershell
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
```

**Binaries (typical paths):**

- Windows: `build\bin\Release\llama-server.exe`, `llama-cli.exe`
- Linux/macOS: `build/bin/llama-server`, `build/bin/llama-cli`

### Smoke test

```bash
./build/bin/llama-server --version
# or: .\build\bin\Release\llama-server.exe --version
```

---

## 4. Run with TurboQuant KV (quick sanity)

Example (adjust model path):

```bash
# Layer-adaptive mode 1 (good default)
TURBO_LAYER_ADAPTIVE=1 ./build/bin/llama-cli -m model.gguf -ctk turbo3 -ctv turbo3 -fa on
```

On Windows (cmd):

```bat
set TURBO_LAYER_ADAPTIVE=1
build\bin\Release\llama-cli.exe -m model.gguf -ctk turbo3 -ctv turbo3 -fa on
```

More context: **[README.md](../README.md)** (TurboQuant section) and **[FORK-CHANGES.md](FORK-CHANGES.md)**.

---

## 5. Optional: Python (conversion scripts, GGUF tooling)

The repo root **[pyproject.toml](../pyproject.toml)** declares **Python ≥ 3.10** (required by current `transformers` / tooling pins).

### Option A — Poetry (full script env, matches `poetry.lock`)

```bash
pip install poetry
poetry install
```

Run entry points via Poetry, e.g.:

```bash
poetry run python convert_hf_to_gguf.py --help
```

### Option B — pip only (minimal / targeted installs)

Use the **`requirements/*.txt`** files that match your task, for example:

- Legacy conversion: `requirements/requirements-convert_legacy_llama.txt`
- Server tests: `tools/server/tests/requirements.txt`
- Tool bench: `requirements/requirements-tool_bench.txt`
- Snapdragon QDC: `scripts/snapdragon/qdc/requirements.txt`

Example:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements/requirements-convert_legacy_llama.txt
```

**GGUF library** (local package): `gguf-py/` — installed as part of `poetry install` or `pip install -e ./gguf-py` when needed.

---

## 6. Optional: rebuild the server Web UI (static assets)

Prebuilt assets may already live under **`tools/server/public/`**. To rebuild from source:

- **Node.js** 18+ (20+ recommended), **npm** 9+
- See **`tools/server/webui/README.md`** for `npm install` and `npm run build`

**Windows note:** the npm **`build`** script ends with a **`post-build.sh`** step. If that step fails in **cmd/PowerShell**, use **Git Bash** or **WSL**, or run the Vite build and copy artifacts as described in `tools/server/webui/README.md`. The Vite build itself usually completes; the shell snippet is what may fail on Windows.

---

## 7. Optional: multimodal (mtmd)

```bash
pip install -r tools/mtmd/requirements.txt
```

Details: **`docs/multimodal.md`** and multimodal subdocs.

---

## 8. What you do *not* need for inference alone

- **Poetry / Node / pip** — not required to compile or run **`llama-server`** if you only use **prebuilt** Web UI assets under `tools/server/public/` and a **GGUF** model on disk.
- **Upstream `docs/install.md`** — describes **prebuilt** packages (winget, brew), not building this fork from source.

---

## 9. Related docs

| Doc | Purpose |
|-----|---------|
| **[MAINTAINING-FORK.md](MAINTAINING-FORK.md)** | Merge **upstream**, canonical `build/`, remotes |
| **[FORK-CHANGES.md](FORK-CHANGES.md)** | What differs from stock llama.cpp |
| **[build.md](build.md)** | All backends, extra CMake options |
| **[OPERATOR-RUNTIME.md](OPERATOR-RUNTIME.md)** | Local launchers, `models.ini`, client snippets outside the repo |
