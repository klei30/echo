# Echo Current Features

Last updated from the current codebase on 2026-05-06.

Echo is a mobile-first personal learning system. The mobile app is the daily surface; Home Brain is the private desktop runtime that powers stronger model, memory, tools, voice, and training.

## Product Loops

| Loop | Built Features |
| --- | --- |
| Daily growth | Talk, Current Read, Today priority, practice reps, check-ins, outcomes |
| Proof and opportunity | Proof items, proof-from-outcome, opportunity generation, missing proof tracking |
| Personalization | training pairs, preference signals, Gemma 4 LoRA training, eval, adapter hot-swap |
| Runtime continuity | Home Brain, Echo Cloud, This Device offline, memory export, queued offline sync |
| Decision support | Decision Room endpoints for perspectives, personal comparison, scenario simulation |
| Agent workflows | Echo MCP product tools for brief, read, decision, memory, signal, proof, opportunities, training |

## Backend Surfaces

| Domain | Representative Endpoints |
| --- | --- |
| Chat/runtime | `/v1/chat/completions`, `/context`, `/save`, `/v1/models` |
| Today/practice | `/v1/today/priority`, `/v1/today/mission`, `/v1/practice/today`, `/v1/practice/log`, `/v1/outcome` |
| Current read | `/v1/thesis/current`, `/v1/user/signal`, `/v1/user/stats`, `/v1/user/rank` |
| Proof | `/v1/proof/items`, `/v1/proof/from-outcome`, `/v1/proof/seed` |
| Opportunities | `/v1/opportunities`, `/v1/opportunities/generate` |
| Memory/rules | `/v1/user/memories`, `/v1/user/rules`, `/v1/user/skills` |
| Training | `/v1/training/status`, `/v1/training/summary`, `/v1/training/runs`, `/v1/training/eval`, `/trigger-training`, `/swap-adapter` |
| Gemma 4 | `/v1/experimental/gemma4/health`, `/v1/experimental/gemma4/chat`, vLLM on `:8003` |
| Offline | `/v1/offline/export`, mobile queued pairs flushed to `/save` |
| Voice | `/v1/voice/token`, LiveKit, voice agent |
| Threads/interventions | `/v1/threads`, `/v1/interventions/next`, `/v1/interventions/ack` |

## Mobile App Surfaces

| Surface | Role |
| --- | --- |
| Talk | Main conversation surface with Echo context, runtime routing, MCP support, voice, and offline Gemma |
| Today | Daily priority, mission, practice, check-in, intervention, outcome capture |
| Passport | Current read, proof/opportunity, progress evidence, training readiness, runtime status |
| Proof | Proof Builder and Opportunities, first-class on desktop and surfaced through Passport on mobile |
| Improve Echo | Training, memory, connections, signals |
| Home Brain | Service health, Gemma 4 vLLM, adapter, tunnel, phone pairing |
| Runtime | Home Brain, Echo Cloud, This Device, LiteRT-LM model download/import, memory sync |
| Tools | Echo MCP workflows and raw MCP server setup |

## Model And Training

- Home Brain runs Gemma 4 E2B through vLLM.
- Echo trains personal LoRA adapters from chat pairs, preference signals, practice outcomes, check-ins, and life events.
- Training uses the desktop/cloud pipeline; mobile offline does not train locally yet.
- This Device uses LiteRT-LM `.litertlm` models and a synced memory pack.
- Offline conversations are queued and uploaded back into `/save` for future training when connected.

## MCP Tools

Public product workflows:

- `echo_training_center`
- `echo_daily_brief`
- `echo_current_read`
- `echo_decision_room`
- `echo_memory_editor`
- `echo_signal_capture`
- `echo_threads_inbox`
- `echo_proof`
- `echo_opportunities`

Lower-level tools remain for advanced/debug compatibility.
