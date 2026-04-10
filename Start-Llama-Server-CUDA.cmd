@echo off
setlocal
title llama-server (TurboQuant — CUDA)  — close this window to STOP
cd /d "%~dp0"

set "HF_HUB_CACHE=%~dp0hf-hub-empty"
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

set "LLAMA_EXE=%~dp0build\bin\Release\llama-server.exe"
if not exist "%LLAMA_EXE%" (
  echo Missing: %LLAMA_EXE%
  echo.
  echo Build first:  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  echo   -DCMAKE_CUDA_ARCHITECTURES=86 ^&^& cmake --build build --config Release
  echo.
  pause
  exit /b 1
)

echo http://127.0.0.1:11436  ^(OpenAI-compatible /v1^)
echo Ollama can stay on :11434
echo.
echo Closing this window stops the server.
echo.

"%LLAMA_EXE%" --models-preset "%~dp0models-cuda.ini" --models-max 1 --host 127.0.0.1 --port 11436

echo.
echo Server exited.
pause
