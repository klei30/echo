# Echo Kaggle Notebook

Main file:

- `echo_gemma4_good_demo.ipynb`

## How It Runs

The notebook uses a real Echo runtime. Set `ECHO_BASE_URL` to a reachable Home Brain, or let the notebook bootstrap the public repo, vLLM Gemma 4, and Echo inside Kaggle.

Optional environment variables:

```bash
ECHO_BASE_URL=https://your-echo-demo-url
ECHO_TOKEN=optional-existing-jwt
ECHO_DEMO_SEED_TOKEN=optional-demo-seed-token
ECHO_GITHUB_REPO=https://github.com/klei30/echo
```

If `ECHO_TOKEN` is not set and `ECHO_DEMO_SEED_TOKEN` is set, the notebook calls `POST /v1/demo/seed` and uses the returned demo JWT. If neither token is set, it registers a temporary empty demo user.

## Real Runtime Requirement

For Kaggle, the notebook should run top-to-bottom against real Gemma 4 and Echo:

```bash
ECHO_GITHUB_REPO=https://github.com/klei30/echo
GEMMA4_MODEL_PATH=/kaggle/input/<your-gemma4-model-folder>
GEMMA4_MAX_MODEL_LEN=8192
ECHO_INSTALL_DEPS=1
ECHO_RUN_TRAINING=1
```

Expected behavior:

- It starts or connects to vLLM Gemma 4.
- It starts or connects to Echo FastAPI.
- It shows Proof Camera, Gemma tool decision, proof creation, opportunity readiness, offline export, privacy boundary, bounded real Unsloth LoRA training, Shadow Clone/LoRA surfaces, and final capability report.
- It does not require your local Home Brain or a private token.
- It does require a Kaggle GPU runtime and an attached Gemma 4 model path for real training, unless `ECHO_BASE_URL` points to a hosted Home Brain.

Live mode is optional and should be used for the video/live demo:

```bash
ECHO_BASE_URL=https://your-echo-demo-url
ECHO_DEMO_SEED_TOKEN=private-token
ECHO_GITHUB_REPO=https://github.com/klei30/echo
```

## GitHub Recommendation

You do not need the public GitHub repo before building the notebook. You should have it public before final Kaggle submission because judges usually need inspectable code, architecture, and reproducibility.

Before pushing or sharing:

- Move secret-like values out of `config.py` and into `.env`.
- Verify `.env`, DB files, logs, adapters, and training data are not tracked.
- Do not publish live tunnel URLs that use default `JWT_SECRET` or `ECHO_SECRET`.
- Update notebook placeholders for video and live demo links.
- Keep `ECHO_DEMO_SEED_TOKEN` private. It enables repeatable public demo data.

## Best Demo Path

The notebook intentionally shows the highest-signal product flow:

1. Runtime negotiation: Home Brain, Cloud Echo, This Device.
2. First-read onboarding.
3. Seeded public-safe learner journey.
4. Proof Camera: Gemma 4 extracts evidence from a real-world artifact payload.
5. Tool decision: Gemma 4 chooses an Echo action and executes it.
6. Gemma-first chat.
7. Today mission and practice.
8. Current read and reality check.
9. Proof creation.
10. Before/after opportunity readiness.
11. Decision Room.
12. Training readiness.
13. Offline export.
14. Privacy boundary: private memory pack vs public Proof Card.

## Final Submission Gaps to Close

Before submitting to Kaggle:

- Replace `TODO` links in the notebook with the final Kaggle notebook, video, live demo, and screenshots/media links.
- Push a cleaned public GitHub repo.
- Add a short video showing the same flow as the notebook.
- Keep the real-runtime bootstrap working, and keep a hosted `ECHO_BASE_URL` as backup if Kaggle startup is slow.
- In public copy, describe Echo as a local-first opportunity engine, not a chatbot, clone app, or productivity tool.
- Claim the notebook runs a bounded real Unsloth LoRA demo loop by default when Kaggle has GPU/model/dependencies. Keep the longer production clone tournament framed as optional.
- If possible, wire the Flutter camera flow to `POST /v1/vision/analyze` and show it in the video.
- If possible, show Home Brain pairing/tunnel continuity and This Device offline continuity in the video.

## Live Demo Endpoints

These are now available for live notebook runs:

- `POST /v1/demo/seed` creates a protected, repeatable Proof Camera demo user.
- `POST /v1/vision/analyze` analyzes artifact text/image inputs into proof-ready JSON, through Gemma and Echo validation.
- `POST /v1/proof/from-artifact` analyzes an artifact and saves it as proof.
- `GET /v1/tools/schema` exposes the Echo tool schema.
- `POST /v1/gemma/tool-call` asks Gemma to choose one Echo action and optionally execute it.

## Current Missing Live Pieces

The notebook now exercises real routes. Remaining polish before final submission:

- Real mobile camera upload in the Flutter app to send `image_url` or `image_base64` into `/v1/vision/analyze`.
- End-to-end validation with the exact Gemma 4 model mount used for submission.
- Confirm `/v1/training/demo-loop` completes within Kaggle time limits; use `/trigger-training` only when you intentionally want the full production clone tournament.
- A video showing Home Brain pairing and This Device offline behavior.
