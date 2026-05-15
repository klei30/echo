@echo off
REM Start experimental Gemma 4 E2B vLLM lane in WSL.
REM Stable Qwen remains on port 8001; Gemma runs on port 8003.

wsl -d Ubuntu-24.04 bash /mnt/c/Users/ASUS/Desktop/echo/start_gemma4_e2b_vllm.sh
