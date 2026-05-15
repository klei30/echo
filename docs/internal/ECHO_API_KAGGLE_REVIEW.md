# Echo API and Kaggle Notebook Review

Review date: 2026-05-12  
Scope: static review of `main.py`, `auth/router.py`, `models/schemas.py`, `config.py`, `README.md`, and `FEATURES.md`.

## Executive Summary

Echo already has enough backend surface to make a strong Gemma 4 Good notebook. The best story is not "chatbot with memory." It is a local-first personal growth loop:

1. Discover direction from first signals and conversations.
2. Turn direction into daily practice and check-ins.
3. Convert behavior into proof.
4. Score opportunities from proof gaps.
5. Improve the user's Gemma 4 Home Brain with feedback, preferences, and adapter training.
6. Export a compact offline pack for This Device / LiteRT continuity.

For the Kaggle notebook, use only the polished product endpoints. Hide debug, admin, raw adapter, and destructive endpoints. Add a small demo seed/reset path or fixtures so judges can run the notebook even if the Home Brain service is not reachable from Kaggle.

## Auth Model

Most endpoints require auth. Public endpoints are:

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Public health check. |
| GET | `/v1/models` | Public OpenAI-compatible model list. Currently returns only `shadow`. |
| POST | `/auth/register` | Public registration, returns JWT. |
| POST | `/auth/login` | Public login, returns JWT. |
| GET | `/docs` | Public FastAPI docs. |
| GET | `/openapi.json` | Public OpenAPI spec. |
| GET | `/redoc` | Public ReDoc docs. |

Authenticated requests accept either:

| Mechanism | How it works | Review |
| --- | --- | --- |
| JWT | `Authorization: Bearer <token>` from `/auth/register` or `/auth/login`. | Best path for notebook and public demo. |
| Static Echo secret | `Authorization: Bearer <ECHO_SECRET>` plus optional `x-echo-user-id`. | Useful for local Home Brain, risky for public tunnel if left as default. |
| Local legacy header | `x-echo-user-id` works only when `ECHO_SECRET` is the default dev secret. | Local dev only. Do not use in public notebook. |

Security issues to fix before public exposure:

- `/save` and `/context` trust the request body `user_id` instead of deriving user identity from `request.state.user_id`.
- `/swap-adapter` accepts a `user_id` body and can hot-swap another user's adapter.
- Debug memory endpoints should be admin-only or disabled outside local dev.
- Default `JWT_SECRET` and `ECHO_SECRET` are dev values. The notebook must not point judges at a tunnel using defaults.
- `config.py` contains a secret-like FCM value. Move that to `.env` before publishing.
- CORS allows all origins. Acceptable for local dev, not ideal for a public tunnel.

## Endpoint Inventory

Legend:

- `Notebook`: include in the Kaggle notebook.
- `Optional`: useful appendix or secondary demo.
- `Hide`: do not show judges unless debugging locally.
- `Fix first`: include only after a small cleanup.

### Auth

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| POST | `/auth/register` | Creates user with email, username, password, optional existing user id claim; returns JWT. | Notebook |
| POST | `/auth/login` | Validates email/password; returns JWT. | Optional |
| GET | `/auth/me` | Returns authenticated user profile. | Notebook |

### Runtime, Chat, Voice

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| GET | `/health` | Basic service health and version. | Notebook |
| GET | `/v1/models` | OpenAI-style model list. Currently only advertises `shadow`. | Fix first |
| POST | `/v1/chat/completions` | Main OpenAI-compatible chat endpoint. Injects memory and loop context, routes to Gemma 4 vLLM / adapter when available, falls back to sparse teacher policy, streams optionally, saves turns for training. | Notebook |
| POST | `/context` | Returns memory + loop system injection, recommended model, LoRA id, confidence, and loop state for a message. | Fix first |
| POST | `/save` | Saves a user/assistant pair into training pairs, confidence, topics, optional memory, events, outcomes, thesis, and returns loop delta. | Fix first |
| POST | `/v1/voice/token` | Returns LiveKit JWT, room, and URL for voice session. | Optional |
| GET | `/v1/runtime/capabilities` | Negotiates Home Brain, Cloud Echo, and This Device capabilities; includes runtimes, features, counts, training summary, and limits. | Notebook |
| GET | `/v1/system/health` | Operational health for database, vLLM, adapter, LiveKit, latest training run. | Notebook |
| GET | `/v1/offline/export` | Exports compact memory/rules/skills/loop/recent events pack for on-device Gemma / LiteRT offline mode. | Notebook |

### Event Stream and Proactive Surface

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| GET | `/v1/events/taxonomy` | Returns product event taxonomy and default actions. | Optional |
| GET | `/v1/events/recent` | Returns recent Echo events with payloads and taxonomy. | Notebook |
| GET | `/v1/events/stream` | SSE stream for proactive mobile surfaces. | Optional |
| GET | `/v1/interventions/next` | Gets or creates next allowed proactive nudge with trust rules and settings. | Notebook |
| POST | `/v1/interventions/ack` | Acknowledges/dismisses an intervention by id. | Optional |
| GET | `/v1/interventions/settings` | Reads intervention settings. | Optional |
| POST | `/v1/interventions/settings` | Updates intervention settings. | Optional |
| POST | `/v1/life/events` | Ingests opt-in real-world events with domain/type/source/privacy metadata. | Notebook |

### Loop, Today, Practice, Check-In

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| GET | `/v1/loop/snapshot` | Summarizes current loop: events, outcomes, latest tournament, training state. | Notebook |
| GET | `/v1/today/priority` | Returns the single most useful next Today card. | Notebook |
| GET | `/v1/today/mission` | Returns priority plus practice, clone, reality, and proof context. | Notebook |
| GET | `/v1/reality/check` | Compares stated intent against behavioral evidence. | Notebook |
| GET | `/v1/practice/today` | Generates or returns cached daily behavioral rep with observation, title, instruction, arc, completion state. | Notebook |
| POST | `/v1/practice/log` | Marks rep done/skipped, records event, outcome, life event, refreshes thesis. | Notebook |
| GET | `/v1/daily/questions` | Generates 3 personalized evening check-in questions from recent messages and rules. | Notebook |
| GET | `/v1/daily/checkin/status` | Returns whether today's check-in is done and streak/readiness value. | Notebook |
| POST | `/v1/daily/checkin` | Saves Q&A, creates background training pairs/memories, returns synthesis bullets. | Notebook |
| POST | `/v1/outcome` | Generic feedback/outcome endpoint. Can mark chat feedback, record outcome, refresh thesis, unlock opportunities, and return loop delta. | Notebook |
| POST | `/v1/reply/action` | Action strip after an Echo reply: save memory, make practice, log outcome, or add to passport/proof. | Optional |

### Current Read, Reflection, Talent

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| GET | `/v1/thesis/current` | Durable current read: belief, evidence, confidence, next test. | Notebook |
| GET | `/v1/growth/timeline` | Longitudinal evidence that the loop is creating change. | Notebook |
| GET | `/v1/revelation/status` | Readiness gate for deeper talent/revelation moment; records product event if ready. | Optional |
| GET | `/v1/user/onboarding-state` | Cold-start stage: day0, early, building, active. | Notebook |
| POST | `/v1/onboarding/first-read` | Saves first onboarding answer, creates immediate first read, outcome, event, thesis refresh. | Notebook |
| GET | `/v1/user/signal` | One sentence capturing who the user is right now from active rules. | Notebook |
| GET | `/v1/user/report` | Single call for Reflect tab: totals, rules, recent messages, weekly reflection. | Optional |
| POST | `/v1/mirror/weekly` | Generates weekly reflection from the last 7 days. | Optional |
| GET | `/v1/user/insights` | Nightly training-style summary: turns analyzed, new patterns, latest insight. | Optional |
| POST | `/v1/emergence` | Generates cross-topic emergence insight if enough data exists. | Optional |
| POST | `/v1/user/talent` | Generates hidden talent narrative from patterns, rules, topics, and messages. | Optional |
| GET | `/v1/user/notable-quote` | Finds a memorable/high-perplexity user statement. | Optional |
| POST | `/v1/user/experiment` | Generates a personalized 7-day behavioral experiment. | Notebook |
| GET | `/v1/user/stats` | Conversation count, weeks active, last trained date, pattern count. | Notebook |
| GET | `/v1/user/rank` | Growth stage and XP derived from pairs, battles, checkpoints, practice. | Optional |
| GET | `/v1/user/confidence` | Topic confidence scores. | Optional |
| GET | `/v1/passport/growth-card` | Privacy-safe shareable growth card: direction, signals, proof count, shareable proof. | Notebook |

### Proof and Opportunities

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| GET | `/v1/proof/items` | Lists active proof items and summary by category/opportunity/outcome. | Notebook |
| POST | `/v1/proof/items` | Creates proof item, records event and life event, refreshes thesis, checks opportunity unlock. | Notebook |
| DELETE | `/v1/proof/items/{item_id}` | Soft-deletes a proof item. | Optional |
| POST | `/v1/proof/from-outcome` | Creates a proof item from an outcome-style body. | Notebook |
| GET | `/v1/opportunities` | Lists saved and suggested opportunities scored against proof gaps. | Notebook |
| POST | `/v1/opportunities` | Creates an opportunity goal with required proof and next step. | Optional |
| POST | `/v1/opportunities/generate` | Generates and saves one opportunity from current read and proof. | Notebook |
| POST | `/v1/proof/seed` | Seeds proof items from history and thesis evidence. | Optional |

Stale docs: `README.md` lists `GET /v1/proof/summary`, but no route exists. Either implement it as a wrapper around `_proof_summary(user_id)` or remove it from the README.

### Memory, Rules, Skills

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| GET | `/v1/user/memories` | Lists mem0 memories for authenticated user. | Notebook |
| POST | `/v1/user/memories` | Directly adds a memory. | Optional |
| POST | `/v1/memory/propose` | Consent-based memory save with privacy level and life event. | Notebook |
| DELETE | `/v1/user/memories` | Deletes all user memories. | Hide |
| DELETE | `/v1/user/memories/{memory_id}` | Deletes one memory. | Hide |
| DELETE | `/v1/debug/memories/{memory_id}` | Admin/debug memory delete by id. | Hide |
| GET | `/v1/debug/memories` | Debug memory list and search. | Hide |
| GET | `/v1/user/rules` | Lists active behavioral/preference rules. | Notebook |
| POST | `/v1/user/rules` | Adds a rule manually. | Optional |
| DELETE | `/v1/user/rules/{rule_id}` | Deletes a rule. | Hide |
| GET | `/v1/user/skills` | Lists extracted active skills. | Notebook |
| POST | `/v1/skills/extract` | Runs skill extraction immediately. | Optional |

### Training, Gemma, Adapters, Teacher

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| GET | `/v1/training/status` | Current training status, latest run, summary for lane. | Notebook |
| GET | `/v1/training/summary` | Readiness from pairs, DPO pairs, outcomes, adapter/runtime status. | Notebook |
| GET | `/v1/training/runs` | Recent training run history with eval score/pass. | Notebook |
| GET | `/v1/training/eval` | Latest adapter evaluation result. | Notebook |
| GET | `/v1/training/history` | Checkpoint creation history. | Optional |
| GET | `/v1/user/history` | Last 50 conversation pairs. | Optional |
| POST | `/trigger-training` | Starts background Gemma 4 LoRA training if data/runtime ready. Stops/restarts vLLM, trains variants or fallback, evaluates, hot-swaps, marks pairs used. | Optional |
| POST | `/swap-adapter` | Direct test endpoint to hot-swap adapter into vLLM. | Hide |
| GET | `/v1/experimental/gemma4/health` | Gemma 4 vLLM lane health. | Notebook |
| POST | `/v1/experimental/gemma4/chat` | Manual Gemma 4 smoke test against vLLM. | Optional |
| GET | `/v1/teacher/policy` | Teacher fallback budget and recent teacher uses. | Notebook |

### Decision Room and Presence Engine

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| POST | `/v1/council/ask` | Runs question through 5 Gemma-powered personas and returns voices plus verdict. | Notebook |
| POST | `/v1/twin/ask` | Compares personal clone vs teacher anonymously; requires trained adapter. | Optional |
| POST | `/v1/twin/choose` | Records which twin answer felt more like the user; saves DPO preference pairs. | Optional |
| POST | `/v1/tournament/run` | Four-style shadow clone tournament on one situation. | Notebook |
| POST | `/v1/tournament/choose` | Saves selected candidate as DPO-ready preference data. | Notebook |
| POST | `/v1/echo/decide` | Thread-aware timing engine: silence, interruption, council, or revelation based on repeated patterns and decision questions. | Notebook |
| GET | `/v1/threads` | Lists active/resolved behavioral threads. | Notebook |
| POST | `/v1/threads/{thread_id}/resolve` | Manually resolves a thread. | Optional |
| POST | `/v1/threads/deduplicate` | Resolves duplicate/stale level-1 threads. | Hide |
| POST | `/v1/echo/simulate` | Parallel Self simulation: current path vs avoided path from behavioral patterns. | Notebook |
| GET | `/v1/clone-mission/latest` | Latest action returned by a winning shadow clone. | Optional |

### Notifications

| Method | Path | What it does | Notebook |
| --- | --- | --- | --- |
| POST | `/v1/user/fcm-token` | Stores device FCM token for push notifications. | Hide |

## What Is Missing for the Best Kaggle Demo

### Must Fix

1. Add a demo seed/reset endpoint or notebook fixture mode.
   - Best API: `POST /v1/demo/seed` creates a synthetic user profile, conversations, proof, outcomes, check-ins, and training run metadata.
   - Alternative: notebook ships `fixtures/echo_demo_responses.json` and runs in "offline fixture mode" if `ECHO_BASE_URL` is unavailable.

2. Fix identity safety for `/save` and `/context`.
   - Use `request.state.user_id` for JWT/static-auth users.
   - Keep body `user_id` only for local admin mode if explicitly allowed.

3. Hide or guard risky endpoints.
   - Gate `/swap-adapter`, debug memory deletes, and delete-all memory behind admin config.
   - Keep `/trigger-training` visible but require explicit confirmation in product UI, not automatic notebook calls.

4. Implement or remove `/v1/proof/summary`.
   - The README advertises it, but code does not implement it.

5. Update `/v1/models`.
   - Should list `gemma4_e2b`, `gemma4_user_<id>` when adapter is available, and `teacher` if cloud fallback is configured.

### Strongly Recommended

1. Add `GET /v1/demo/notebook-state`.
   - One call returning the exact assets the notebook needs: user, capability table, loop snapshot, thesis, mission, proof, opportunities, training, offline export sample.

2. Add `POST /v1/vision/analyze` or `POST /v1/life/events/from-image`.
   - Current API is text-first. The hackathon emphasizes multimodal understanding. Echo can frame this as: mobile captures a real-world proof artifact, Gemma reads it, and Echo turns it into proof/opportunity.

3. Add a native tool/function calling demo.
   - Current tool path routes away from local Gemma when a tool prompt is detected.
   - For the notebook, expose Echo APIs as callable tools and show Gemma selecting `create_proof`, `log_outcome`, or `generate_opportunity`.

4. Add deterministic sample outputs.
   - Some endpoints need 5, 10, or 20 stored moments before they become impressive.
   - The notebook should seed enough data to avoid "still learning" responses.

5. Add public-safe privacy copy.
   - Growth Card and offline export are strong, but the notebook should explicitly show that raw private conversations are not in the shareable card.

## Recommended Kaggle Notebook Story

Notebook title:

`Echo Home Brain: A Local Gemma 4 Mentor That Turns Private Signals Into Proof and Opportunity`

Target tracks:

- Main Track: full-stack app, backend, local runtime, training, mobile/desktop architecture.
- Digital Equity: private mentor for users without elite access, low connectivity, local compute.
- Future of Education: personalized practice and evidence loop.
- LiteRT / edge: offline export pack for This Device.
- Unsloth: personal LoRA training and eval-gated hot swap.
- Cactus/mobile: transparent routing between Home Brain, Cloud Echo, and This Device.

### Notebook Structure

1. **Why Echo exists**
   - One short markdown cell: mentor scarcity, privacy, weak connectivity, and proof/opportunity gap.

2. **Connect to Echo**
   - Set `ECHO_BASE_URL`.
   - Register/login demo user.
   - Health check: `/health`, `/auth/me`, `/v1/system/health`.
   - If unavailable, switch to fixture mode.

3. **Runtime negotiation**
   - Call `/v1/runtime/capabilities`.
   - Render table: Home Brain, Cloud Echo, This Device.
   - Show what works offline vs online.

4. **First signal onboarding**
   - Call `/v1/onboarding/first-read`.
   - Then `/v1/user/onboarding-state` and `/v1/thesis/current`.
   - Show how one answer becomes a first read and training signal.

5. **Seed a realistic learner journey**
   - Use `/save`, `/v1/life/events`, `/v1/memory/propose`, and `/v1/outcome`.
   - Persona: student/founder in low-connectivity environment building a portfolio without mentors.
   - Keep data synthetic and public-safe.

6. **Main chat with Echo**
   - Call `/v1/chat/completions`.
   - Show OpenAI-compatible payload.
   - Display model used and note Gemma-first routing.

7. **Today loop**
   - Call `/v1/today/mission`, `/v1/practice/today`, `/v1/daily/questions`.
   - Log practice with `/v1/practice/log`.
   - Submit check-in with `/v1/daily/checkin`.

8. **Current read and evidence**
   - Call `/v1/thesis/current`, `/v1/reality/check`, `/v1/growth/timeline`, `/v1/user/signal`.
   - Show the difference between "profile" and "evidence-backed read."

9. **Proof engine**
   - Create proof with `/v1/proof/items`.
   - Convert outcome to proof with `/v1/proof/from-outcome`.
   - Show `/v1/passport/growth-card` as public-safe output.

10. **Opportunity engine**
   - Call `/v1/opportunities` and `/v1/opportunities/generate`.
   - Render readiness, missing proof, next step.
   - This is the most important "impact" cell.

11. **Decision Room**
   - Call `/v1/council/ask`.
   - Call `/v1/tournament/run` and `/v1/tournament/choose`.
   - Call `/v1/echo/simulate`.
   - Optional: `/v1/twin/ask` only if adapter exists.

12. **Improve Echo**
   - Call `/v1/training/summary`, `/v1/training/runs`, `/v1/training/eval`, `/v1/teacher/policy`.
   - Do not trigger training automatically in Kaggle.
   - If Home Brain is ready and user explicitly opts in, show `/trigger-training`.

13. **Offline continuity**
   - Call `/v1/offline/export`.
   - Show compressed memory, rules, skills, loop state, recent pairs/check-ins.
   - Explain how This Device can keep working without the desktop.

14. **Final impact report**
   - Combine thesis, practice, proof, opportunity, runtime, and training status into one final dataframe/report.
   - End with "what changed for the user" not only "what endpoints returned."

15. **Appendix: endpoint explorer**
   - Lightweight helper to call optional endpoints.
   - Hide debug/admin endpoints by default.

## Notebook API Helper Design

Use a tiny client, not a framework:

```python
import os, json, time, requests

BASE_URL = os.getenv("ECHO_BASE_URL", "http://127.0.0.1:8002").rstrip("/")
TOKEN = os.getenv("ECHO_TOKEN")

def echo(method, path, **kwargs):
    headers = kwargs.pop("headers", {})
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    response = requests.request(method, f"{BASE_URL}{path}", headers=headers, timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()
```

Notebook should prefer JWT:

```python
auth = requests.post(
    f"{BASE_URL}/auth/register",
    json={"email": f"demo-{int(time.time())}@echo.local", "username": "kaggle_demo", "password": "demo-password"},
    timeout=30,
).json()
TOKEN = auth["token"]
```

## Competition Notes

The official Kaggle page is JavaScript-rendered and did not expose readable details through the crawler during this review. Public sources available on 2026-05-12 describe the Gemma 4 Good Hackathon as a Google DeepMind/Kaggle challenge focused on real-world impact, local/edge intelligence, native tool use, multimodal understanding, and tracks including health/science, education, global resilience, digital equity, safety/trust, LiteRT, Cactus, Ollama, llama.cpp, and Unsloth.

Before final submission, manually verify the live rules at:

- https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview
- https://www.kaggle.com/competitions/gemma-4-good-hackathon/rules

Secondary sources used for planning:

- https://www.hackathons.space/hackathons/the-gemma-4-good-hackathon
- https://internshala.com/competitions/the-gemma-4-good-hackathon/
- https://www.linkedin.com/company/kaggle
- https://www.competehub.dev/en/competitions/kagglegemma-4-good-hackathon

## Final Recommendation

For launch, build the notebook around 18 core calls:

1. `/auth/register`
2. `/auth/me`
3. `/health`
4. `/v1/runtime/capabilities`
5. `/v1/system/health`
6. `/v1/onboarding/first-read`
7. `/save`
8. `/v1/chat/completions`
9. `/v1/today/mission`
10. `/v1/practice/today`
11. `/v1/practice/log`
12. `/v1/thesis/current`
13. `/v1/proof/items`
14. `/v1/opportunities`
15. `/v1/council/ask`
16. `/v1/echo/simulate`
17. `/v1/training/summary`
18. `/v1/offline/export`

That is enough to show Echo's power without overwhelming judges.
