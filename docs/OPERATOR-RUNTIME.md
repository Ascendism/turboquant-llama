# Operator runtime folder (your local “server kit”)

There are **two different things** people often mix up:

| Layer | What it is | Where it lives |
|--------|------------|----------------|
| **A. Forked `llama-server` binary** | Build output from this repo (CUDA, TurboQuant, any merged `tools/server/` C++ changes). | `build/bin/Release/` after CMake; source under **`tools/server/`** in **this repo** (published as [turboquant-llama](https://github.com/CarapaceUDE/turboquant-llama)). |
| **B. Local launch kit** | Scripts + **`models.ini`** + client JSON snippets + logs/cache dirs. **Machine-specific paths**, ports, and integration with *your* apps. | e.g. **`%USERPROFILE%\llama-turboquant-runtime`** on your PC. **Not** required for the fork to be useful to others. |

Throughout maintenance work we treat **layer A** (git + build) as authoritative. **Layer B** lives outside the repo unless you choose to version it.

---

## Typical layout (example)

- **`Start-Llama-Server.cmd`**: sets a dummy **`HF_HUB_CACHE`**, points at **`C:\app\llama-cpp-turboquant-cuda\build\bin\Release\llama-server.exe`**, runs **`--models-preset`**, **`--models-max`**, host/port (e.g. **`11436`**).
- **`models.ini`**: upstream **`--models-preset`** file ([`tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) with **`[*]`** defaults and per-model sections. Paths are **private** to your machine.
- **`client-provider-snippet.json`**: example **client** config (provider URL + model ids) for apps that use OpenAI-compatible HTTP—adjust for your integration.

**Shortcut:** e.g. **`C:\app\Start-Llama-Server.lnk`** → **`Start-Llama-Server.cmd`** in the runtime folder.

None of that **replaces** building from this repo; it **configures** the fork binary on your desk.

---

## What to expose “to the outside world”

**Already public:** this fork’s **`llama-server`** and server source.

**Templates:** **[examples/models-preset/](../examples/models-preset/)** (`models.ini.example`). Do not publish real paths or internal client secrets.

---

## Relationship to [MAINTAINING-FORK.md](MAINTAINING-FORK.md)

After each upstream merge: rebuild **`build/`**, then your launcher keeps working as long as **`LLAMA_EXE`** still points at `build\bin\Release\llama-server.exe` and **`models.ini`** stays valid.
