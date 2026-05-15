import asyncio
import json
import uuid
from datetime import datetime as dt

import httpx

from config import settings
from db.database import get_conn
from loop_events import record_event, record_life_event, record_outcome
from proactive_engine import create_clone_mission, get_latest_clone_mission
from router.confidence import detect_topic, get_confidence
from teacher_policy import infer_importance, record_teacher_usage, should_use_teacher
from thesis import refresh_current_thesis
from training.adapter import adapter_exists, ensure_adapter_loaded, lora_name_for_user
from training.collector import save_pair


STYLES: dict[str, str] = {
    "Strategist": (
        "You are the Strategist clone. Help by turning the situation into a clear plan, "
        "tradeoff, or next move. Be specific and grounded."
    ),
    "Challenger": (
        "You are the Challenger clone. Help by naming the avoidance, contradiction, or "
        "weak assumption the user may be missing. Be direct but not cruel."
    ),
    "Mirror": (
        "You are the Mirror clone. Help by reflecting the user's deeper pattern back with "
        "evidence and emotional precision. Avoid generic encouragement."
    ),
    "Builder": (
        "You are the Builder clone. Help by converting the situation into a system, rule, "
        "constraint, or repeatable practice."
    ),
}


MENTOR_PERSONA = (
    "You are the outside Mentor challenger. You are not the user's main Echo. "
    "Your job is to provide one unusually strong alternative answer that can teach "
    "the local model what better looks like. Address the user directly, never address "
    "Gemma or another model. Avoid generic advice. Be concise, specific, and grounded."
)


def _practice_for_style(style: str, prompt: str, response: str) -> dict:
    title_by_style = {
        "Strategist": "Choose the next move",
        "Challenger": "Challenge the assumption",
        "Mirror": "Name the pattern",
        "Builder": "Build the rep",
    }
    instruction_by_style = {
        "Strategist": "Before the next similar moment ends, write the one concrete move you will take and do the smallest version of it.",
        "Challenger": "When this pattern appears today, name the assumption you are protecting and test the opposite for five minutes.",
        "Mirror": "Pause when the pattern appears, say what is really happening in one sentence, then choose the next honest move.",
        "Builder": "Turn the winning answer into one repeatable action and complete it once today.",
    }
    observation = "Shadow tournament winner: " + response[:260].replace("\n", " ").strip()
    if len(observation) > 320:
        observation = observation[:319].rstrip() + "..."
    return {
        "observation": observation or f"The {style} clone won this situation.",
        "rep_title": title_by_style.get(style, "Practice the winner"),
        "rep_instruction": instruction_by_style.get(style, "Practice the winning shadow response once today."),
        "arc_label": f"Tournament: {style}",
    }


async def _create_practice_from_winner(user_id: str, style: str, prompt: str, response: str) -> str | None:
    practice = _practice_for_style(style, prompt, response)
    rep_id = str(uuid.uuid4())
    today = dt.utcnow().strftime("%Y-%m-%d")
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO practice_reps
                (id, user_id, date, observation, rep_title, rep_instruction, arc_label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET
                observation=excluded.observation,
                rep_title=excluded.rep_title,
                rep_instruction=excluded.rep_instruction,
                arc_label=excluded.arc_label,
                created_at=datetime('now')
            """,
            (
                rep_id,
                user_id,
                today,
                practice["observation"],
                practice["rep_title"],
                practice["rep_instruction"],
                practice["arc_label"],
            ),
        )
        await db.commit()
        async with db.execute(
            "SELECT id FROM practice_reps WHERE user_id=? AND date=?",
            (user_id, today),
        ) as cur:
            row = await cur.fetchone()
    return row["id"] if row else None


async def _call_gemma_clone(user_id: str, prompt: str, style: str, persona: str) -> tuple[str, str]:
    use_adapter = False
    if adapter_exists(user_id, lane="gemma4_e2b"):
        use_adapter = await ensure_adapter_loaded(user_id, lane="gemma4_e2b")
    model = lora_name_for_user(user_id, lane="gemma4_e2b") if use_adapter else settings.gemma4_base_model
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.gemma4_vllm_base_url.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.85,
                "max_tokens": 260,
                "messages": [
                    {"role": "system", "content": persona},
                    {
                        "role": "user",
                        "content": (
                            "Respond as one candidate shadow clone. Keep it useful, personal, "
                            "and under 180 words.\n\nUser situation:\n" + prompt
                        ),
                    },
                ],
                "stop": ["<function", "<tool_call>", "<|tool_call|>", "✿FUNCTION✿"],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip(), model


async def _call_teacher(prompt: str, style: str, persona: str) -> str:
    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            f"{settings.teacher_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.teacher_model,
                "temperature": 0.65,
                "max_tokens": 260,
                "messages": [
                    {"role": "system", "content": persona},
                    {
                        "role": "user",
                        "content": (
                            "Respond as one candidate shadow clone. Keep it useful, personal, "
                            "and under 180 words.\n\nUser situation:\n" + prompt
                        ),
                    },
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def create_tournament(user_id: str, prompt: str) -> dict:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt required")

    run_id = str(uuid.uuid4())
    topic = detect_topic(prompt)
    confidence = await get_confidence(user_id, prompt)
    teacher_decision = await should_use_teacher(
        user_id,
        "tournament_challenger",
        confidence=confidence,
        importance=infer_importance(prompt),
        prompt=prompt,
    )
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO tournament_runs (id, user_id, prompt, topic, status)
            VALUES (?, ?, ?, ?, 'candidate')
            """,
            (run_id, user_id, prompt, topic),
        )
        await db.commit()

    gemma_results = await asyncio.gather(
        *[_call_gemma_clone(user_id, prompt, style, persona) for style, persona in STYLES.items()],
        return_exceptions=True,
    )
    teacher_result = None
    if teacher_decision.allowed:
        try:
            teacher_result = await _call_teacher(prompt, "Mentor", MENTOR_PERSONA)
            await record_teacher_usage(
                user_id,
                "tournament_challenger",
                teacher_decision.reason,
                {"run_id": run_id, "topic": topic, "confidence": confidence, "decision": teacher_decision.to_dict()},
            )
        except Exception as e:
            teacher_result = e

    candidates = []
    async with get_conn() as db:
        for (style, _), result in zip(STYLES.items(), gemma_results):
            model_used = settings.gemma4_base_model
            if isinstance(result, Exception):
                response = ""
            else:
                response, model_used = result
            if not response:
                response = f"{style} could not generate a candidate this round."
            candidate_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO tournament_candidates (id, run_id, style, response, score, signals_json)
                VALUES (?, ?, ?, ?, 0.0, ?)
                """,
                (candidate_id, run_id, style, response, json.dumps({"initial": True, "model": model_used})),
            )
            candidates.append({"id": candidate_id, "style": style, "response": response, "score": 0.0})
        if teacher_decision.allowed:
            response = "" if isinstance(teacher_result, Exception) else str(teacher_result or "")
            if response:
                candidate_id = str(uuid.uuid4())
                await db.execute(
                    """
                    INSERT INTO tournament_candidates (id, run_id, style, response, score, signals_json)
                    VALUES (?, ?, ?, ?, 0.0, ?)
                    """,
                    (
                        candidate_id,
                        run_id,
                        "Mentor",
                        response,
                        json.dumps({"initial": True, "model": settings.teacher_model, "teacher": True}),
                    ),
                )
                candidates.append({"id": candidate_id, "style": "Mentor", "response": response, "score": 0.0})
        await db.commit()

    await record_event(
        user_id,
        "tournament_created",
        "tournament",
        f"{len(candidates)} shadow candidates competed on a {topic} situation.",
        {
            "run_id": run_id,
            "topic": topic,
            "styles": [c["style"] for c in candidates],
            "teacher_policy": teacher_decision.to_dict(),
        },
        weight=1.5,
    )
    return {
        "run_id": run_id,
        "topic": topic,
        "candidates": candidates,
        "teacher_policy": teacher_decision.to_dict(),
    }


async def choose_candidate(user_id: str, run_id: str, candidate_id: str, outcome: str = "chosen") -> dict:
    async with get_conn() as db:
        async with db.execute(
            "SELECT * FROM tournament_runs WHERE id=? AND user_id=?",
            (run_id, user_id),
        ) as cur:
            run = await cur.fetchone()
        if not run:
            raise LookupError("tournament not found")

        if run["status"] == "complete" and run["winning_style"]:
            mission = await get_latest_clone_mission(user_id)
            return {
                "saved": True,
                "already_chosen": True,
                "winning_style": run["winning_style"],
                "run_id": run_id,
                "practice_rep_id": mission.get("practice_rep_id") if mission and mission.get("run_id") == run_id else None,
                "clone_mission": mission if mission and mission.get("run_id") == run_id else None,
                "learning_summary": "This tournament was already chosen, so Echo returned the existing result.",
            }

        async with db.execute(
            "SELECT * FROM tournament_candidates WHERE id=? AND run_id=?",
            (candidate_id, run_id),
        ) as cur:
            chosen = await cur.fetchone()
        if not chosen:
            raise LookupError("candidate not found")

        async with db.execute(
            "SELECT * FROM tournament_candidates WHERE run_id=? AND id<>? ORDER BY created_at ASC",
            (run_id, candidate_id),
        ) as cur:
            rejected = await cur.fetchone()

        await db.execute(
            "UPDATE tournament_candidates SET score=score+1.0, signals_json=? WHERE id=?",
            (json.dumps({"outcome": outcome}), candidate_id),
        )
        await db.execute(
            "UPDATE tournament_runs SET status='complete', winning_style=? WHERE id=?",
            (chosen["style"], run_id),
        )
        await db.commit()

    await save_pair(
        user_id=user_id,
        user_msg=run["prompt"],
        assistant_msg=chosen["response"],
        model_used=f"tournament:{chosen['style']}",
        engagement_signal="thumbs_up",
        topic=run["topic"],
    )
    if rejected:
        await save_pair(
            user_id=user_id,
            user_msg=run["prompt"],
            assistant_msg=rejected["response"],
            model_used=f"tournament:{rejected['style']}",
            engagement_signal="thumbs_down",
            topic=run["topic"],
        )

    event_id = await record_event(
        user_id,
        "tournament_winner",
        "tournament",
        f"{chosen['style']} won the latest shadow clone tournament.",
        {"run_id": run_id, "candidate_id": candidate_id, "style": chosen["style"], "topic": run["topic"]},
        weight=2.0,
    )
    await record_outcome(
        user_id,
        subject_type="tournament_candidate",
        subject_id=candidate_id,
        outcome=outcome,
        score=1.0,
        event_id=event_id,
        note=f"User selected {chosen['style']}.",
    )
    practice_rep_id = await _create_practice_from_winner(
        user_id,
        chosen["style"],
        run["prompt"],
        chosen["response"],
    )
    if practice_rep_id:
        await record_event(
            user_id,
            "practice_created",
            "tournament",
            f"Created a practice rep from the {chosen['style']} tournament winner.",
            {"run_id": run_id, "rep_id": practice_rep_id, "style": chosen["style"]},
            weight=1.4,
        )
    mission = await create_clone_mission(
        user_id=user_id,
        run_id=run_id,
        candidate_id=candidate_id,
        winning_style=chosen["style"],
        response=chosen["response"],
        practice_rep_id=practice_rep_id,
    )
    await record_life_event(
        user_id,
        "clone",
        "clone_mission_created",
        "tournament",
        "A clone returned with an action.",
        mission["suggested_action"],
        {"run_id": run_id, "mission_id": mission["id"], "style": chosen["style"]},
        confidence=0.8,
        subject_type="clone_mission",
        subject_id=mission["id"],
    )
    await refresh_current_thesis(user_id)
    return {
        "saved": True,
        "winning_style": chosen["style"],
        "run_id": run_id,
        "practice_rep_id": practice_rep_id,
        "clone_mission": mission,
        "learning_summary": (
            f"Echo learned that {chosen['style']} helped most for this {run['topic']} situation. "
            "That choice now updates the current read, creates a practice rep, returns a clone mission, and becomes preference data for training."
        ),
    }
