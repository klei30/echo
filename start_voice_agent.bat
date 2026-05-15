@echo off
title Echo Voice Agent
cd /d "%~dp0"

echo ==========================================
echo   Echo Voice Agent - Starting...
echo ==========================================
echo.
echo LiveKit:  ws://localhost:7880
echo API Key:  devkey
echo.

:retry
python voice_agent.py start ^
  --url ws://localhost:7880 ^
  --api-key devkey ^
  --api-secret secret

echo.
echo [!] Voice agent stopped. Restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto retry
