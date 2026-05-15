# Echo + vLLM + ChatMCP Setup Guide

Complete guide to start and run the entire Shadow Clone pipeline end-to-end.

## Prerequisites

- Windows 11 with WSL2 (Ubuntu-24.04)
- NVIDIA GPU with CUDA support
- Android SDK + Android Emulator
- Flutter SDK
- Python 3.13+

## Architecture Overview

```
ChatMCP (Android Emulator)
    ↓ (OpenAI-compatible API)
Echo FastAPI (port 8002, Windows)
    ├─→ mem0 + Qdrant (local vector DB)
    ├─→ SQLite (training pairs)
    └─→ vLLM (WSL port 8001)
         └─→ Qwen2.5-7B-Instruct + LoRA adapters
```

---

## Step 1: Start vLLM in WSL

vLLM must run first and stay running. It requires the `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` env var to enable hot-swap of LoRA adapters.

### One-time setup (first time only)
```bash
wsl -d Ubuntu-24.04 bash -c "
pip install -q vllm torch
"
```

### Start vLLM (every session)
```bash
wsl -d Ubuntu-24.04 bash -c "
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
nohup /home/klei/vllm-env/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --max-lora-rank 64 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  >> /tmp/vllm.log 2>&1 &
"
```

**Verify vLLM is running:**
```bash
wsl -d Ubuntu-24.04 bash -c "curl -s http://localhost:8001/health"
```

Expected output: `(empty or HTTP response)` — if it hangs, vLLM is still loading the model (~30-60s first time).

---

## Step 2: Start Echo (Windows)

Echo is the FastAPI sidecar that routes between ChatMCP, memory, and vLLM.

### Navigate to echo directory
```bash
cd C:\Users\ASUS\Desktop\echo
```

### Start Echo
```bash
python main.py
```

Echo will:
1. Initialize SQLite database (if first run)
2. Warm up mem0 + Qdrant
3. Start APScheduler for nightly training
4. Listen on `http://localhost:8002`

**Verify Echo is running:**
```bash
curl -s http://localhost:8002/health
```

Expected output:
```json
{"status":"ok","version":"2.0"}
```

---

## Step 3: Start Android Emulator

### Launch the emulator
```bash
flutter emulators --launch Pixel_10_Pro
```

Wait for the lock screen to appear (~30 seconds first time).

**Verify emulator is ready:**
```bash
adb devices
```

Expected output:
```
List of devices attached
emulator-5554	device
```

---

## Step 4: Run ChatMCP on Emulator

### Build and install ChatMCP APK
```bash
cd C:\Users\ASUS\Desktop\ChatMCP
flutter run -d emulator-5554
```

This will:
1. Compile the Flutter app
2. Install the APK on the emulator
3. Launch ChatMCP and hot-reload server

The app should open automatically on the emulator. If it doesn't, manually launch from the app drawer.

---

## Step 5: Configure ChatMCP Echo Provider (One-time)

In the ChatMCP app:

1. **Settings** → **LLM Settings** → **Custom Provider**
2. **Provider Name:** Echo
3. **Base URL:** `http://10.0.2.2:8002/v1`
4. **API Key:** `dummy` (Echo doesn't validate it)
5. **Model:** `shadow`
6. Save

The magic IP `10.0.2.2` from the emulator routes to your Windows host's `localhost`.

---

## Step 6: Test End-to-End Routing

### Send a message in ChatMCP

Open ChatMCP, type: `"What do I do for work?"`

The response should mention your profession (freelance backend developer) — this comes from memory injection + your personal LoRA model.

### Verify routing in Echo logs

```bash
tail -f C:\Users\ASUS\Desktop\echo\main.py
# Or from the running Echo window
```

Look for:
```
INFO:echo:/v1/chat/completions user=0abcba6b-... confidence=0.85 route=gemma
```

- **`confidence=0.85`** means the model has enough data to use your personal adapter
- **`route=gemma`** means it's routing to your local Qwen model (not OpenAI)

If you see `route=openai`, confidence is below 0.70 — you need more training pairs first.

---

## Monitoring & Debugging

### Watch Echo logs
```bash
# From PowerShell in echo directory
Get-Content main.py -Wait | Select-String "route=|confidence="
```

### Check vLLM logs
```bash
wsl -d Ubuntu-24.04 bash -c "tail -f /tmp/vllm.log"
```

### Verify LoRA adapters loaded in vLLM
```bash
wsl -d Ubuntu-24.04 bash -c "curl -s http://localhost:8001/v1/models" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for m in d['data']: print(m['id'])
"
```

Expected (after training):
```
Qwen/Qwen2.5-7B-Instruct
user_default
user_0abcba6b-2a4a-4a66-951c-e5e6a68f1da3
```

### Check confidence scores in SQLite
```bash
cd C:\Users\ASUS\Desktop\echo
python3 -c "
import sqlite3
conn = sqlite3.connect('echo.db')
rows = conn.execute('SELECT user_id, topic, score FROM confidence ORDER BY score DESC LIMIT 10').fetchall()
for r in rows: print(f'{r[0][:20]:20} {r[1]:12} {r[2]:.2f}')
"
```

---

## Troubleshooting

### vLLM not responding
- Check it's running: `wsl -d Ubuntu-24.04 bash -c "ps aux | grep vllm"`
- Verify env var: `wsl -d Ubuntu-24.04 bash -c "ps aux | grep vllm | grep VLLM_ALLOW"`
- If missing, restart vLLM with the env var set (see Step 1)

### Echo returning OpenAI responses
- Check confidence: should be ≥0.70 to route to local model
- Send more messages to build up confidence
- Check logs: `tail -f main.py` should show routing decision

### ChatMCP stuck/frozen
- Force stop: `adb -s emulator-5554 shell am force-stop run.daodao.chatmcp`
- Relaunch: manually open from app drawer or `adb -s emulator-5554 shell am start -S run.daodao.chatmcp/.MainActivity`
- If emulator is frozen: Restart the emulator entirely

### LoRA adapter not loading
- Verify vLLM has `--enable-lora --max-lora-rank 64` flags
- Check adapter files exist: `ls C:\Users\ASUS\Desktop\echo\adapters\`
- Verify Echo logs show `Hot-swapped LoRA` message after training

### No responses from vLLM
- Check GPU memory: `nvidia-smi` (should show 14+ GB used)
- Verify network: `wsl -d Ubuntu-24.04 bash -c "curl -s http://localhost:8001/health"`
- Check vLLM logs for errors: `tail -50 /tmp/vllm.log`

---

## Nightly Training Schedule

Echo runs automatic training every night at 2am UTC (if set up):

```bash
# Check scheduled jobs
python3 -c "
from apscheduler.schedulers.background import BackgroundScheduler
# Jobs are set in echo/scheduler.py
"
```

Training requires:
1. **≥20 training pairs** collected from chat
2. **vLLM running** with LoRA env var
3. **WSL Ubuntu-24.04** available (for LlamaFactory)

---

## Full Startup Sequence (Quick Reference)

```bash
# Terminal 1: WSL vLLM
wsl -d Ubuntu-24.04 bash -c "
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
/home/klei/vllm-env/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora --max-lora-rank 64 --port 8001 \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --host 0.0.0.0
"

# Terminal 2: Echo (Windows)
cd C:\Users\ASUS\Desktop\echo
python main.py

# Terminal 3: Android Emulator
flutter emulators --launch Pixel_10_Pro

# Terminal 4: ChatMCP (after emulator is ready)
cd C:\Users\ASUS\Desktop\ChatMCP
flutter run -d emulator-5554
```

Once all 4 are running, ChatMCP will route to your personal Qwen model when confidence ≥0.70.

---

## What's Happening Under the Hood

1. **ChatMCP** sends message to Echo's `/v1/chat/completions`
2. **Echo** fetches your user ID from ChatMCP's persistent storage
3. **Echo** calls `/context` to get memory injection + routing hint
4. **Confidence router** checks if your score ≥0.70 for that topic
5. **If local**: Echo calls **vLLM** with your personal LoRA adapter
6. **If remote**: Echo calls **OpenAI** (teacher model)
7. **Save pairs**: Echo stores user/assistant messages for future training
8. **Memory injection**: Echo adds system prompt with user context from mem0
9. **Nightly scheduler**: Echo triggers LlamaFactory SFT training on collected pairs
10. **Hot-swap**: After training, Echo loads the new LoRA adapter into vLLM

---

## Key Configuration Files

- `C:\Users\ASUS\Desktop\echo\config.py` — Settings (confidence threshold, model paths, etc.)
- `C:\Users\ASUS\Desktop\echo\main.py` — FastAPI routes
- `C:\Users\ASUS\Desktop\echo\router\confidence.py` — Confidence scoring logic
- `C:\Users\ASUS\Desktop\echo\training\orchestrator.py` — LlamaFactory integration
- `C:\Users\ASUS\Desktop\echo\training\adapter.py` — Hot-swap logic
- `C:\Users\ASUS\Desktop\ChatMCP\lib\echo\echo_client.dart` — ChatMCP Echo integration
- `C:\Users\ASUS\Desktop\ChatMCP\lib\page\layout\chat_page\chat_page.dart` — Routing logic

---

## Next Steps

1. ✅ Verify all 4 components start cleanly
2. ✅ Test routing: send a message, check logs for `route=gemma`
3. ✅ Collect training data: chat naturally for 20+ turns
4. ✅ Monitor nightly training (2am UTC) or trigger manually: `curl -X POST http://localhost:8002/trigger-training -H "Content-Type: application/json" -d '{"user_id":"0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"}'`
5. ✅ Check confidence increases after training
6. ✅ Implement DPO training (requires thumbs up/down feedback)
