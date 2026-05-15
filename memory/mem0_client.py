import asyncio
import logging
import traceback
import os
import re
import time
from mem0 import Memory
from config import settings

log = logging.getLogger("echo.mem0")
_memory: Memory | None = None
_user_has_memories: set[str] = set()
_core_cache: dict[str, tuple[float, list[str]]] = {}


_CUSTOM_UPDATE_PROMPT = """
You are managing a personal memory store for a user. Compare new information with existing memories and decide what to do.

Rules for conflict resolution:
- If new info CONTRADICTS an existing memory (e.g., different city, different date, different preference), use UPDATE to replace the old memory with the new one. Never keep both conflicting facts.
- If new info is the SAME as an existing memory, use NONE (skip it).
- If new info ADDS new detail to an existing memory (e.g., adds a destination to a trip that had none), use UPDATE.
- Only use ADD for genuinely new facts not related to any existing memory.
- Use DELETE if the user explicitly says something is no longer true.

Memory categories to track:
- Personal facts (name, location, job, family)
- Plans and dates (vacations, meetings, goals) — always UPDATE when destination or date changes
- Preferences (communication style, food, hobbies)
- Projects and work
- Behavioral patterns

Output JSON list of operations: ADD, UPDATE, DELETE, or NONE.
"""

_CUSTOM_INSTRUCTIONS = (
    "Extract only facts that matter for future personalization. "
    "For location-based plans (vacations, trips), always capture the specific destination and date. "
    "When the user corrects or updates a previous fact, prefer UPDATE over ADD. "
    "Do not extract generic conversational phrases or questions — only concrete facts about the user."
)


def _memory_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _message_text(messages: list[dict], role: str = "user") -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != role:
            continue
        content = msg.get("content", "")
        parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


def _clean_memory_candidate(text: str, max_len: int = 260) -> str | None:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = cleaned.strip(" `\"'")
    cleaned = re.sub(r"^(?:that|this)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:please|thanks|thank you)\.?$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" `\"'")
    if not cleaned or len(cleaned) < 4 or len(cleaned) > max_len:
        return None
    if "<" in cleaned or ">" in cleaned or cleaned.endswith("?"):
        return None
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return cleaned


def _user_fact_sentence(fact: str) -> str | None:
    cleaned = _clean_memory_candidate(fact)
    if not cleaned:
        return None
    lower = cleaned.lower()
    replacements = (
        ("my ", "User's "),
        ("i am ", "User is "),
        ("i'm ", "User is "),
        ("i work ", "User works "),
        ("i prefer ", "User prefers "),
        ("i like ", "User likes "),
        ("i dislike ", "User dislikes "),
        ("i don't like ", "User does not like "),
        ("i dont like ", "User does not like "),
        ("i want ", "User wants "),
    )
    for prefix, replacement in replacements:
        if lower.startswith(prefix):
            return replacement + cleaned[len(prefix):]
    return cleaned[0].upper() + cleaned[1:]


def _explicit_memories_from_messages(messages: list[dict]) -> list[str]:
    text = _message_text(messages, role="user")
    if not text:
        return []

    candidates: list[str] = []

    for match in re.finditer(
        r"\bremember(?:\s+this\s+about\s+me|\s+that|\s+this)?\s*:?\s+(?P<fact>[^\n]{4,260})",
        text,
        flags=re.IGNORECASE,
    ):
        fact = _user_fact_sentence(match.group("fact"))
        if fact:
            candidates.append(fact)

    for match in re.finditer(
        r"\bmy\s+(?:real\s+)?name\s+is\s+(?P<name>[A-Za-z][A-Za-z .'-]{1,60})",
        text,
        flags=re.IGNORECASE,
    ):
        name = _clean_memory_candidate(match.group("name"), max_len=80)
        if name:
            candidates.append(f"User's real name is {name.rstrip('.')}.")

    for match in re.finditer(
        r"\bcall\s+me\s+(?P<name>[A-Za-z][A-Za-z .'-]{1,60})",
        text,
        flags=re.IGNORECASE,
    ):
        name = _clean_memory_candidate(match.group("name"), max_len=80)
        if name:
            candidates.append(f"User prefers to be called {name.rstrip('.')}.")

    for match in re.finditer(
        r"\bforget\s+(?P<name>[A-Za-z][A-Za-z0-9 .'-]{1,60})",
        text,
        flags=re.IGNORECASE,
    ):
        name = _clean_memory_candidate(match.group("name"), max_len=80)
        if name:
            candidates.append(f"User asked Echo to forget {name.rstrip('.')}.")

    profile_patterns = (
        (
            r"\bi\s+work\s+(?:at|for|with)\s+(?P<value>[A-Za-z0-9][A-Za-z0-9 .&'-]{1,100})",
            "User works at {value}.",
        ),
        (
            r"\bi\s+live\s+in\s+(?P<value>[A-Za-z][A-Za-z .'-]{1,80})",
            "User lives in {value}.",
        ),
        (
            r"\bi\s+am\s+from\s+(?P<value>[A-Za-z][A-Za-z .'-]{1,80})",
            "User is from {value}.",
        ),
        (
            r"\bi(?:'m| am)\s+(?:a|an)\s+(?P<value>[A-Za-z][A-Za-z0-9 .&'-]{2,100})",
            "User is a {value}.",
        ),
        (
            r"\bmy\s+(?:main\s+)?goal\s+is\s+(?P<value>[A-Za-z0-9][A-Za-z0-9 .,&'-]{2,140})",
            "User's goal is {value}.",
        ),
        (
            r"\bmy\s+priority\s+(?:today|right now|now)?\s*is\s+(?P<value>[A-Za-z0-9][A-Za-z0-9 .,&'-]{2,140})",
            "User's priority is {value}.",
        ),
        (
            r"\bmy\s+favorite\s+(?P<field>[A-Za-z ]{2,40})\s+is\s+(?P<value>[A-Za-z0-9][A-Za-z0-9 .,&'-]{1,100})",
            "User's favorite {field} is {value}.",
        ),
    )
    for pattern, template in profile_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            values = {k: (v or "").strip(" .,'\"") for k, v in match.groupdict().items()}
            sentence = _clean_memory_candidate(template.format(**values))
            if sentence:
                candidates.append(sentence)

    preference_match = re.search(
        r"\b(?:i\s+prefer|i\s+like\s+when|i\s+don't\s+like|i\s+dont\s+like|please\s+don't|please\s+dont|never|always|from\s+now\s+on)\b",
        text,
        flags=re.IGNORECASE,
    )
    if preference_match:
        preference = _clean_memory_candidate(text, max_len=220)
        if preference:
            candidates.append(f"User preference: {preference}")

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _memory_key(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique[:3]


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        qdrant_path = os.path.abspath("./qdrant_data")
        os.makedirs(qdrant_path, exist_ok=True)
        if settings.google_api_key and "googleapis.com" in settings.teacher_base_url:
            # The OpenAI-compatible Gemini shim can ignore/loosen JSON mode.
            # Native Gemini gives mem0 a real application/json response contract.
            llm = {
                "provider": "gemini",
                "config": {
                    "model": settings.teacher_model,
                    "api_key": settings.google_api_key,
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "max_tokens": 1200,
                },
            }
        else:
            llm_config: dict = {
                "model": settings.teacher_model,
                "api_key": settings.llm_api_key,
                "temperature": 0.0,
                "top_p": 0.1,
                "max_tokens": 1200,
            }
            if "openai.com" not in settings.teacher_base_url:
                llm_config["openai_base_url"] = settings.teacher_base_url
            llm = {
                "provider": "openai",
                "config": llm_config,
            }

        config = {
            "llm": llm,
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5",
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": qdrant_path,
                    "embedding_model_dims": 384,
                },
            },
            "custom_update_memory_prompt": _CUSTOM_UPDATE_PROMPT,
            "custom_instructions": _CUSTOM_INSTRUCTIONS,
        }
        _memory = Memory.from_config(config)
    return _memory


async def warmup() -> None:
    try:
        _get_memory()
        provider = "Gemini" if settings.google_api_key and "googleapis.com" in settings.teacher_base_url else "OpenAI-compatible"
        log.info("mem0 initialized (%s LLM + fastembed local embedder)", provider)
    except Exception as e:
        log.warning("mem0 warmup failed: %s", e)


async def search_memories(query: str, user_id: str, limit: int = 12) -> list[str]:
    try:
        results = await asyncio.to_thread(
            _get_memory().search, query, filters={"user_id": user_id}, top_k=limit
        )
        if isinstance(results, list):
            return [r["memory"] for r in results if r.get("memory")]
        return [r["memory"] for r in results.get("results", []) if r.get("memory")]
    except Exception as e:
        log.warning("mem0 search failed: %s", e)
        return []


async def get_all_memories(user_id: str, limit: int = 1000) -> list[dict]:
    try:
        result = await asyncio.to_thread(
            _get_memory().get_all, filters={"user_id": user_id}, top_k=limit
        )
        return result if isinstance(result, list) else result.get("results", [])
    except Exception as e:
        log.warning("mem0 get_all failed: %s", e)
        return []


async def get_core_memories(user_id: str, ttl_seconds: int = 30) -> list[str]:
    now = time.monotonic()
    cached = _core_cache.get(user_id)
    if cached and now - cached[0] < ttl_seconds:
        return cached[1]

    items = await get_all_memories(user_id)
    keywords = (
        "real name", "preferred name", "full name", "name is",
        "work", "works", "job", "dialogo", "founder", "engineer",
        "preference", "prefers", "conversation", "bullet", "tone",
    )
    core: list[str] = []
    for item in items:
        text = item.get("memory") or item.get("text") or ""
        if not text:
            continue
        lower = text.lower()
        if any(k in lower for k in keywords):
            core.append(text)
    _core_cache[user_id] = (now, core[:24])
    return core[:24]


async def _store_explicit_fallback_memories(messages: list[dict], user_id: str, reason: str) -> int:
    memories = _explicit_memories_from_messages(messages)
    if not memories:
        return 0

    existing = await get_all_memories(user_id)
    existing_keys = {
        _memory_key(item.get("memory") or item.get("text") or "")
        for item in existing
    }

    saved = 0
    for memory in memories:
        key = _memory_key(memory)
        if not key or key in existing_keys:
            continue
        if await add_raw_memory(memory, user_id=user_id, source=f"fallback_{reason}"):
            saved += 1
            existing_keys.add(key)

    if saved:
        log.info("mem0 fallback saved=%d reason=%s user=%s", saved, reason, user_id)
    return saved


async def add_memories(messages: list[dict], user_id: str) -> None:
    try:
        result = await asyncio.to_thread(
            _get_memory().add,
            messages,
            user_id=user_id,
            metadata={"source": "echo_auto"},
            prompt=_CUSTOM_INSTRUCTIONS,
        )
        items = result if isinstance(result, list) else result.get("results", [])
        added = [r["memory"] for r in items if r.get("event") == "ADD"]
        updated = [r["memory"] for r in items if r.get("event") == "UPDATE"]
        if added or updated:
            _user_has_memories.add(user_id)
            _core_cache.pop(user_id, None)
        log.info("mem0 ADD=%d UPDATE=%d for user=%s", len(added), len(updated), user_id)
        if not added and not updated:
            await _store_explicit_fallback_memories(messages, user_id, reason="zero_result")
    except Exception as e:
        log.warning("mem0 add failed for user=%s: %s\n%s", user_id, e, traceback.format_exc())
        await _store_explicit_fallback_memories(messages, user_id, reason="error")


async def add_raw_memory(memory_text: str, user_id: str, source: str = "manual") -> bool:
    text = " ".join((memory_text or "").split())
    if not text:
        return False
    try:
        result = await asyncio.to_thread(
            _get_memory().add,
            [{"role": "user", "content": text}],
            user_id=user_id,
            metadata={"source": source},
            infer=False,
        )
        items = result if isinstance(result, list) else result.get("results", [])
        if items:
            _user_has_memories.add(user_id)
            _core_cache.pop(user_id, None)
        log.info("mem0 RAW_ADD=%d for user=%s", len(items), user_id)
        return bool(items)
    except Exception as e:
        log.warning("mem0 raw add failed for user=%s: %s\n%s", user_id, e, traceback.format_exc())
        return False
