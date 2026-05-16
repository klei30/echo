# Echo Kaggle Submission

Public notebook:

<https://www.kaggle.com/code/kleialiajjj/echo-gemma-4-good-demo>

Source notebook:

- `echo_gemma4_good_demo.ipynb`

## Vision

Echo is a local-first AI proof engine that turns real-world work into
opportunity-ready evidence. The notebook follows Noor, a public-safe learner
persona whose ability is visible in rough artifacts: repair notes, prototype
tests, teaching moments, feedback, and practice logs.

Gemma 4 reads those signals, extracts structured evidence, chooses Echo tool
actions, builds Proof Cards, identifies missing proof, and keeps the next step
available through Home Brain and offline/mobile continuity.

## Run Modes

The notebook is designed for judged execution, not only for a perfect local
machine.

| Mode | How to run | What it proves |
| --- | --- | --- |
| Hosted Echo | Set `ECHO_BASE_URL` to a reachable Home Brain backend | Real Echo API, seeded demo user, runtime negotiation, proof/opportunity flow |
| Kaggle-local Gemma | Attach a Gemma 4 model and set `GEMMA4_MODEL_PATH` or `KAGGLE_GEMMA4_MODEL_PATH` | vLLM + Echo bootstrap, real Gemma runtime, optional bounded training path |
| Presentation fixture | No backend/model required, or set `ECHO_FORCE_PRESENTATION_MODE=1` | Full product story executes with clearly labeled fixture evidence |

The latest Kaggle version completed successfully with `KernelWorkerStatus.COMPLETE`.

## What The Notebook Shows

1. Runtime negotiation: Home Brain, Echo Cloud, This Device, fixture fallback.
2. First-read onboarding for Noor.
3. Public-safe journey seed data.
4. Proof Camera: Gemma-style extraction from a real-world artifact payload.
5. Tool decision: create proof, log outcome, generate opportunity, or request feedback.
6. Before/after opportunity readiness.
7. Daily practice and check-in loop.
8. Current Read, reality check, and growth timeline.
9. Decision Room: council, tournament, and parallel-self reasoning.
10. Training readiness, dataset examples, eval gate, and adapter hot-swap boundary.
11. Offline export for This Device continuity.
12. Final judged evidence dashboard.

## Submission Assets

| Asset | Location |
| --- | --- |
| Writeup draft/support copy | `submission/writeup.md` |
| Link and checklist helper | `submission/attachments.md` |
| Media gallery images | `media_gallery/` |
| Kaggle metadata | `kernel-metadata.json` |

## Final Manual Checks

- Kaggle writeup title/subtitle match `submission/writeup.md`.
- Impact Track is selected.
- GitHub repo link is attached.
- Kaggle notebook link is attached.
- Cover image and video are attached in the Kaggle writeup UI.
- No private `.env`, DB files, logs, adapters, raw training data, or active tunnel URLs are attached.

