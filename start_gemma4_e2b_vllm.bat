@echo off
REM Start experimental Gemma 4 E2B vLLM lane in WSL.
REM Stable Qwen remains on port 8001; Gemma runs on port 8003.

for /f "delims=" %%I in ('wsl -d Ubuntu-24.04 wslpath -u "%~dp0start_gemma4_e2b_vllm.sh"') do set "GEMMA4_SCRIPT=%%I"
if not defined GEMMA4_SCRIPT (
  echo Could not translate the Echo start script to a WSL path.
  exit /b 1
)
wsl -d Ubuntu-24.04 bash "%GEMMA4_SCRIPT%"
