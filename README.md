<p align="center">
  <img src="assets/echo_logo.png" alt="Echo logo" width="120" />
</p>

<h1 align="center">Echo</h1>

<p align="center">
  <strong>A local-first AI proof engine that turns real-world work into opportunity-ready evidence.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC--BY%204.0-lightgrey?style=flat-square" alt="CC-BY 4.0" /></a>
  &nbsp;
  <a href="https://github.com/klei30/echo_mobile"><img src="https://img.shields.io/badge/Mobile%20App-echo__mobile-black?style=flat-square&logo=flutter" alt="Mobile App" /></a>
  &nbsp;
  <a href="https://www.kaggle.com/code/kleialiajjj/echo-gemma-4-good-demo"><img src="https://img.shields.io/badge/Kaggle-Gemma%204%20Good-20BEFF?style=flat-square&logo=kaggle" alt="Kaggle Notebook" /></a>
</p>

<p align="center">
  <a href="https://github.com/klei30/echo">Backend</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/klei30/echo_mobile">Mobile App</a>
  &nbsp;|&nbsp;
  <a href="https://www.kaggle.com/code/kleialiajjj/echo-gemma-4-good-demo">Kaggle Notebook</a>
</p>

---

> Talent is often invisible because the proof is trapped in the wrong format.

Most people are talented. Only some discover it, and often only because someone happened to notice the right signal at the right time.

Echo is built for the people who are missed by that accident: learners, builders, repairers, translators, caregivers, explainers, and self-taught makers whose ability is real but under-documented. Their best evidence may live in rough notes, repairs, screenshots, practice logs, family responsibilities, feedback from others, or a project that worked before it looked polished.

Echo is not an AI mentor or a productivity chatbot. It is local-first proof infrastructure: a private loop that helps a person notice, document, test, and package evidence of ability without turning their private life into somebody else's training data.

Gemma 4 reads real-world effort, extracts the signal, creates private Proof Cards, identifies what proof is still missing, and recommends the next step toward a scholarship, apprenticeship, portfolio, local job, maker grant, or community opportunity.

The loop is deliberately small and repeatable:

1. Capture real signal from what someone already does.
2. Turn that signal into a living map of patterns and proof.
3. Recommend one small next practice step.
4. Record the outcome.
5. Use the result to improve the map, the evidence, and the personal model.

Most AI products answer questions. Echo helps a person build the evidence to be believed, while keeping raw memory private and the user in control of what becomes public proof.

---

## Start Here

| If you are... | Read this first |
| --- | --- |
| Reviewing the project | [Why Echo Exists](#why-echo-exists), [Product Loop](#product-loop), [Gemma 4 Usage](#gemma-4-usage), [Kaggle Demo](#kaggle-demo) |
| Running the backend | [Quick Start](#quick-start), [Backend API](#backend-api) |
| Pairing the mobile app | [Mobile Pairing](#mobile-pairing), [Runtime Model](#runtime-model) |
| Evaluating the architecture | [Home Brain](#home-brain), [Personal Training](#personal-training), [Privacy Boundaries](#privacy-boundaries) |
| Connecting agents | [MCP Workflows](#mcp-workflows) |

## Judging Links

| Asset | Link |
| --- | --- |
| Backend repository | <https://github.com/klei30/echo> |
| Mobile app repository | <https://github.com/klei30/echo_mobile> |
| Kaggle notebook | <https://www.kaggle.com/code/kleialiajjj/echo-gemma-4-good-demo> |
| Notebook source | [kaggle/echo_gemma4_good_demo.ipynb](kaggle/echo_gemma4_good_demo.ipynb) |
| Writeup support files | [kaggle/submission](kaggle/submission) |
| Media gallery assets | [kaggle/media_gallery](kaggle/media_gallery) |

## What Echo Does

Echo is for people whose ability is real but under-documented: students without mentors, builders without polished portfolios, people in low-connectivity places, and anyone whose best evidence lives in rough artifacts instead of official credentials.

- Builds a living **Pattern Map** from conversations, memories, check-ins, outcomes, proof, and repeated behavior.
- Generates one concrete **Next Proof Step** for daily practice.
- Captures rough artifacts as **Proof Cards**: photos, notes, repairs, screenshots, feedback, practice wins, and project evidence.
- Supports **Decision Room** reasoning for real choices, tradeoffs, and missing proof.
- Tracks outcomes instead of stopping at chat.
- Works local-first through a desktop **Home Brain** and an offline-capable mobile runtime.
- Adapts over time through personal LoRA training on the user's own signal.

### Example Flow

```text
Student uploads a repair photo
  -> Echo extracts the skill signal
  -> Echo creates a private Proof Card
  -> Echo spots missing evidence for a portfolio
  -> Echo recommends one small practice rep
  -> The outcome updates the Pattern Map
  -> Future answers and training data improve
```

---

## Why Echo Exists

Most people discover their own talent by accident. A teacher notices something. A project reveals a skill. A random moment creates confidence. For many people, that moment never comes, not because they lack ability, but because their evidence is in the wrong format.

The systems that create opportunity reward people who already know how to document themselves: polished writing, stable internet, public portfolios, confident interviews. But real ability often shows up somewhere else: in repairs, explanations, care work, translations, rough sketches, practice logs, and the way other people rely on someone without either person naming it.

Echo is built around one belief: talent is hidden in ordinary behavior. The things someone keeps returning to, finishes without being asked, improves quietly, or struggles to explain to others are signal. Echo captures that signal privately, turns it into proof, and closes the loop between what someone has already done and where they could go next.

That makes Echo a response to a dignity problem as much as a product problem. AI should not reduce a person to a score, transcript, profile, or collection of extractable data. It should help people keep agency over their own story. Echo separates private memory from public proof so a user can decide what becomes visible, what stays local, and what evidence is strong enough to carry into an opportunity system.

## Product Loop

```text
Signal -> Pattern Map -> Next Proof Step -> Outcome -> Proof Card -> Direction
```

Echo treats growth as a loop, not a one-time recommendation.

### Pattern Map

A living hypothesis about what a person is drawn toward, where they show consistency, what they are improving at, and what evidence is still missing.

It is not a personality quiz. It is built from actual behavior: chats, check-ins, outcomes, proof items, memories, repeated themes, decisions, and practice results.

### Next Proof Step

One small rep grounded in the Pattern Map:

- write one paragraph
- solve one problem
- record one explanation
- publish one rough artifact
- ask for one piece of feedback
- document one repair, lesson, or experiment

The rep creates an outcome. The outcome updates the map.

### Decision Room

Decision Room helps with choices that depend on personal context:

- Should I apply?
- Which path fits me?
- What proof is missing?
- What happens if I keep doing this for six months?
- Which option creates the strongest next evidence?

Modes include **Council**, **Twin**, **Tournament**, and **Parallel Self**. Choices become preference signal for future training runs.

### Proof Cards

Proof Cards turn private effort into reusable evidence:

- project summaries
- artifact descriptions
- practice wins
- feedback quotes
- skills practiced
- outcomes
- public-safe versions for portfolios or applications

The point is not to expose private memory. The point is to help someone package what they have actually done.

### Direction

Echo maps proof against real goals: a portfolio, scholarship, apprenticeship, maker grant, repair log, teaching artifact, local project, or community contribution. Missing proof becomes the next proof step.

---

## Runtime Model

Echo has three runtime layers.

| Runtime | Role |
| --- | --- |
| **Home Brain** | Private desktop runtime: Echo API, SQLite, mem0/Qdrant, Gemma 4 via vLLM, Unsloth/LlamaFactory training, custom LoRAs, eval, MCP tools, LiveKit voice, tunnel pairing |
| **This Device** | Offline phone runtime: LiteRT-LM Gemma, synced memory pack, cached Today state, queued chats and outcomes |
| **Echo Cloud** | Online continuity when Home Brain is unavailable |

Echo is local-first because the people who most need opportunity infrastructure cannot assume stable internet, paid cloud AI, elite networks, or polished credentials.

## Home Brain

Home Brain is the private desktop runtime. It runs the stronger model lane, stores the user's local state, exposes the backend API, coordinates tools, and trains personal adapters.

Home Brain owns:

- `echo.db` for conversations, outcomes, proof, opportunities, and product state
- `qdrant_data/` for vector memory
- `training_data/` for supervised and preference datasets
- `adapters/` for custom LoRA outputs
- vLLM for Gemma 4 inference
- LiveKit voice integration
- MCP workflows for agent access

## Personal Training

Echo adapts by training adapter variants on the user's own signal, then evaluating them before anything live changes.

The model does not learn a generic persona. It learns which evidence, proof steps, decision patterns, and response styles actually helped this user move forward. Training is explicit, local, and eval-gated.

```text
Feedback -> Preference Pairs -> Training Data -> Custom LoRA -> Eval -> Home Brain
```

Training variants:

| Variant | What it trains on |
| --- | --- |
| **SFT** | Best interactions and preferred answer style |
| **SeqKD** | Teacher model reasoning distilled into the Gemma lane |
| **Self-critique** | Corrections from weak or inaccurate outputs |
| **Group DPO** | Preferences from Decision Room choices and comparisons |
| **On-policy** | Improvements sampled against live Echo outputs |

Pipeline:

```text
User signal
  -> training pairs + preference pairs
  -> Unsloth / LlamaFactory
  -> SFT / SeqKD / Self-critique / Group DPO / On-policy
  -> held-out eval
  -> winning LoRA
  -> hot-swap into vLLM
```

The adapter belongs to the user. It is trained on their signal, on their machine, and evaluated before it touches the live runtime.

### Shadow Clone Training

The training architecture takes inspiration from Naruto's Shadow Clone training arc. Naruto uses many clones to practice an extremely difficult technique, the Rasenshuriken, in parallel. Each clone tries, fails, adjusts, and when the clones disappear, their experience returns to the original. The point is not duplication for its own sake. It is accelerated learning through many bounded attempts.

Echo uses that idea as a technical metaphor, not as product branding. The system creates multiple training views from the user's signal, lets each one explore a different learning strategy, evaluates the results, and only merges back what actually improves the Home Brain.

Each training lane acts like a specialized clone:

| Clone lane | What it explores |
| --- | --- |
| **SFT** | Which answers, proof steps, and summaries helped most |
| **SeqKD** | What teacher-model reasoning should be distilled into the Gemma lane |
| **Self-critique** | Where weak outputs should be corrected |
| **Group DPO** | Which Decision Room choices the user preferred |
| **On-policy** | How live Echo responses can improve from real outcomes |

The eval gate decides what comes back. If an adapter does not pass, Echo keeps the previous Home Brain instead of promoting a weaker model.

CPU safety tests cover the orchestration and promotion gate. The bounded
[GPU acceptance test](docs/GPU_ACCEPTANCE_TEST.md) verifies the remaining
Gemma 4, Unsloth, CUDA, and vLLM boundary.

The active production path is `training/coordinator.py` plus
`training/orchestrator.py` and the SQLite state modules. The older
`training/pipeline.py`/`clones.py` Postgres and OpenAI experiment is retained
as reference code and remains disabled by `legacy_openai_pipeline_enabled`.

---

## Gemma 4 Usage

Echo uses Gemma 4 across the product:

- context-aware chat
- Current Read synthesis
- daily practice generation
- Decision Room reasoning
- proof extraction from artifacts and outcomes
- opportunity gap analysis
- offline LiteRT-LM on Android
- Unsloth LoRA training with hot-swap on Home Brain

## Kaggle Demo

The Kaggle notebook demonstrates the Gemma 4 Good flow:

<https://www.kaggle.com/code/kleialiajjj/echo-gemma-4-good-demo>

Source notebook:

[kaggle/echo_gemma4_good_demo.ipynb](kaggle/echo_gemma4_good_demo.ipynb)

It is the fastest way to review the core concept without running the full Home Brain stack. It runs in two judged-safe paths: a real Echo/Gemma runtime when a hosted backend or mounted Gemma 4 model is available, and a clearly labeled presentation fixture when Kaggle has no model input attached.

For full Kaggle-local vLLM mode, the GPU matters. Use T4, L4, A10/A10G, A100, L40/L40S, or newer. Kaggle P100/K80-class GPUs are not reliable with current vLLM CUDA wheels; on those GPUs the notebook can use a real Transformers inference fallback, but full vLLM hot-swap and Unsloth training proof should use a compatible GPU or a hosted `ECHO_BASE_URL`.

---

## Quick Start

### Requirements

- Windows 11
- Python 3.13+
- WSL2 Ubuntu 24.04 for the full Home Brain path
- NVIDIA GPU with CUDA for vLLM and training
- Flutter and Android tooling for the mobile app
- Gemma 4 model files for local inference

### Minimal Backend

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python main.py
```

Check health:

```bash
curl http://localhost:8002/health
curl http://localhost:8002/v1/models
```

By default, the FastAPI backend listens on port `8002`.

### Full Home Brain Stack

Start the full service stack:

```powershell
.\start_echo_services.bat
```

Optional GPU and training dependencies:

```bash
pip install vllm torch
pip install llamafactory
pip install unsloth
```

### Mobile Pairing

Mobile app repository:

<https://github.com/klei30/echo_mobile>

Backend URL:

- Android emulator: `http://10.0.2.2:8002`
- Real device: start backend, start tunnel, show QR, then scan from `You -> Runtime -> Pair Computer`

---

## Backend API

Representative endpoints:

| Domain | Endpoints |
| --- | --- |
| Runtime / chat | `/health`, `/v1/models`, `/v1/chat/completions`, `/context`, `/save`, `/v1/runtime/capabilities` |
| Today / practice | `/v1/today/priority`, `/v1/today/mission`, `/v1/practice/today`, `/v1/practice/log`, `/v1/outcome`, `/v1/daily/checkin` |
| Current Read | `/v1/thesis/current`, `/v1/user/signal`, `/v1/user/stats`, `/v1/growth/timeline` |
| Memory / rules | `/v1/user/memories`, `/v1/memory/propose`, `/v1/user/rules`, `/v1/user/skills` |
| Proof | `/v1/proof/items`, `/v1/proof/from-outcome`, `/v1/proof/from-artifact`, `/v1/proof/seed`, `/v1/vision/analyze` |
| Opportunities | `/v1/opportunities`, `/v1/opportunities/generate` |
| Decision Room | `/v1/echo/decide`, `/v1/council/ask`, `/v1/twin/ask`, `/v1/twin/choose`, `/v1/tournament/run`, `/v1/tournament/choose` |
| Training | `/v1/training/status`, `/v1/training/summary`, `/v1/training/runs`, `/v1/training/eval`, `/trigger-training`, `/swap-adapter` |
| Gemma / tools | `/v1/experimental/gemma4/health`, `/v1/experimental/gemma4/chat`, `/v1/tools/schema`, `/v1/gemma/tool-call` |
| Offline / voice / events | `/v1/offline/export`, `/v1/voice/token`, `/v1/events/recent`, `/v1/events/stream` |

Example request:

```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"echo","messages":[{"role":"user","content":"What should I practice today?"}]}'
```

---

## MCP Workflows

`echo_mcp.py` exposes Echo to MCP-compatible agents.

Public product workflows:

- `echo_daily_brief`
- `echo_current_read`
- `echo_decision_room`
- `echo_proof`
- `echo_opportunities`
- `echo_training_center`
- `echo_memory_editor`
- `echo_signal_capture`
- `echo_threads_inbox`

Example MCP config:

```json
{
  "mcpServers": {
    "echo": {
      "command": "python",
      "args": ["/path/to/echo/echo_mcp.py"],
      "env": {
        "ECHO_USER_ID": "your_user_id",
        "ECHO_SECRET": "echo-local-secret"
      }
    }
  }
}
```

---

## Privacy Boundaries

Echo's strongest mode keeps all personal data on the user's machine:

- `echo.db` stores conversations, outcomes, proof, opportunities, and product state.
- `qdrant_data/` stores vector memory.
- `training_data/` stores training datasets.
- `adapters/` stores custom LoRAs.

Current boundaries:

- Home Brain is required for full Gemma inference, voice, MCP tools, and LoRA training.
- On-device LoRA loading is not yet implemented.
- On-device training is not yet implemented.
- Public tunnels require production secrets and endpoint hardening.

---

## Repository Layout

```text
echo/
  main.py                   FastAPI app, routes, streaming
  config.py                 Environment settings
  auth/                     JWT auth
  db/                       SQLite schema and helpers
  memory/                   mem0 and Qdrant retrieval
  router/                   Topic detection, confidence, routing
  training/                 Data collection, Unsloth/LlamaFactory,
                            SFT/DPO variants, adapter runtime, eval
  thesis.py                 Current Read synthesis
  loop_events.py            Event and outcome recording
  loop_priority.py          Daily priority logic
  proactive_engine.py       Interventions, discovery gate, scheduler
  teacher_policy.py         Sparse fallback gating
  echo_mcp.py               MCP workflow server
  voice_agent.py            LiveKit voice agent
  kaggle/                   Gemma 4 Good notebook and demo assets
```

Mobile app:

<https://github.com/klei30/echo_mobile>

---

## License

CC-BY 4.0. Submitted to the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon).
