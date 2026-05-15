"""
Echo Voice Agent — livekit-agents 1.5.x
STT (Whisper) → Echo sidecar (memory + routing) → TTS (OpenAI alloy)

Start:
    python voice_agent.py start \
        --url ws://localhost:7880 \
        --api-key devkey \
        --api-secret secret
"""
import asyncio
import json as _json
import logging
import os

import httpx
from livekit.agents import Agent, AgentSession, JobContext, AgentServer, cli
from livekit.agents import room_io, TurnHandlingOptions
from livekit.plugins import openai as lk_openai, silero

from config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("echo.voice_agent")

# Make credentials available to plugins and CLI
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")
os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)

server = AgentServer()


def prewarm(proc):
    log.info("Prewarming VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    log.info("VAD ready.")


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    log.info("Voice agent joining room: %s", ctx.room.name)
    await ctx.connect()

    # Extract user identity from room name ("voice-{user_id}")
    user_id = ctx.room.name.removeprefix("voice-") or "default"
    log.info("Voice session for user_id=%s", user_id)

    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

    session = AgentSession(
        stt=lk_openai.STT(model="whisper-1"),
        # Route through Echo sidecar — injects memory context and saves the pair
        llm=lk_openai.LLM(
            model="shadow",
            base_url="http://localhost:8002/v1",
            api_key="voice-agent",
            extra_headers={"x-echo-user-id": user_id},
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=10.0, pool=5.0),
        ),
        tts=lk_openai.TTS(
            voice="alloy",
            api_key=settings.openai_api_key or None,
        ),
        vad=vad,
        turn_handling=TurnHandlingOptions(
            min_endpointing_delay=0.5,
            max_endpointing_delay=2.0,
        ),
    )

    def _relay(data: dict) -> None:
        payload = _json.dumps(data).encode()
        asyncio.ensure_future(ctx.room.local_participant.publish_data(payload))

    def _on_transcript(e):
        if e.is_final:
            log.info("STT [%s]: %r", user_id, e.transcript)
            _relay({"type": "transcript", "role": "user", "text": e.transcript})

    def _on_state(e):
        state_str = str(e.new_state).split(".")[-1].lower()
        log.info("agent %s->%s [%s]", e.old_state, e.new_state, user_id)
        _relay({"type": "agent_state", "state": state_str})

    session.on("user_input_transcribed", _on_transcript)
    session.on("agent_state_changed", _on_state)

    await session.start(
        agent=Agent(
            instructions=(
                "You are Echo, a private AI mentor that helps the user turn hidden potential into real-world opportunity. "
                "Your loop is: discover direction, practice today, build proof, log outcomes, and improve from real signals. "
                "Keep every reply to 1-3 sentences with one concrete next step when useful. "
                "Avoid clone, shadow, battle, tournament, lab, or anime-style language. "
                "Speak naturally — no markdown, asterisks, or special characters."
            ),
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
            audio_output=room_io.AudioOutputOptions(),
        ),
    )

    log.info("Voice agent active in room: %s", ctx.room.name)
    await session.say("Hello! I'm Echo. Go ahead and speak.")


if __name__ == "__main__":
    cli.run_app(server)
