@echo off
setlocal
title llama-server (TurboQuant — Vulkan)  — close this window to STOP
cd /d "%~dp0"

set "HF_HUB_CACHE=%~dp0hf-hub-empty"
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

set "LLAMA_EXE=%~dp0build-vulkan\bin\Release\llama-server.exe"
if not exist "%LLAMA_EXE%" set "LLAMA_EXE=%~dp0build-vulkan\bin\llama-server.exe"
if not exist "%LLAMA_EXE%" (
  echo Could not find llama-server.exe in:
  echo   %~dp0build-vulkan\bin\Release\
  echo   %~dp0build-vulkan\bin\
  echo.
  echo Build first (delete build-vulkan folder if it already exists, then run):
  echo   cmake -B build-vulkan -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00" -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00" ^&^& cmake --build build-vulkan --config Release
  echo.
  pause
  exit /b 1
)

echo http://127.0.0.1:11436  ^(OpenAI-compatible /v1^)
echo.
echo Closing this window stops the server.
echo.

"%LLAMA_EXE%" --models-preset "%~dp0models-vulkan.ini" --models-max 1 --host 127.0.0.1 --port 11436

echo.
echo Server exited.
pause
