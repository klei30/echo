import asyncio
import logging
import re
from memory.mem0_client import get_core_memories, search_memories
from db.database import get_conn

# Imported lazily to avoid circular imports — used only inside _build_injection
def _get_base_prompt() -> str:
    from main import ECHO_PRODUCT_SYSTEM_PROMPT
    return ECHO_PRODUCT_SYSTEM_PROMPT

log = logging.getLogger("echo.context")


def _strip_thoughts(text: str) -> str:
    text = re.sub(r"<thought>.*?</thought>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"```(?:\w+)?", "", text)
    return " ".join(text.replace("```", "").split())


def _clean_rule_text(text: str) -> str | None:
    raw = text or ""
    cleaned = _strip_thoughts(raw)
    if not cleaned or cleaned.upper() == "NO_PREFERENCE":
        return None
    if "NO_PREFERENCE" in cleaned.upper():
        return None
    if "<" in cleaned or ">" in cleaned:
        return None
    if len(cleaned) > 220:
        return None
    real_name = re.search(r"\breal name is\s+([a-z][a-z .'-]{1,60})", cleaned, flags=re.IGNORECASE)
    if real_name and "forget" in cleaned.lower():
        name = real_name.group(1).split(",")[0].strip(" .'-")
        if name:
            return f"Use the user's real name: {name.title()}."
    return cleaned


def _forbidden_names(rules: list[dict]) -> set[str]:
    names: set[str] = set()
    for rule in rules:
        for name in rule.get("_forbidden_names", []):
            if name:
                names.add(str(name).lower())
        text = (rule.get("rule_text") or "").lower()
        for match in re.finditer(r"\bforget\s+([a-z][a-z0-9_-]{1,30})", text):
            names.add(match.group(1))
    return names


def _canonical_name_memory(memory: str) -> str | None:
    match = re.search(r"\b(real name|full name)\s+is\s+([A-Za-z][A-Za-z .'-]{1,80})", memory, flags=re.IGNORECASE)
    if not match:
        return None
    label = match.group(1).lower()
    name = re.split(
        r"\b(?:and|which|instead|previously|but)\b|,",
        match.group(2),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .'-")
    if not name:
        return None
    return f"User's {label} is {name}."


def _normalise_memory(text: str, forbidden_names: set[str]) -> str | None:
    memory = _strip_thoughts(text)
    if not memory or "<" in memory or ">" in memory:
        return None
    lower = memory.lower()
    for name in forbidden_names:
        if not name:
            continue
        has_forbidden_name = re.search(rf"\b{re.escape(name)}\b", lower) is not None
        if not has_forbidden_name:
            continue
        has_correction = "real name" in lower or "instead of" in lower or "forget" in lower
        canonical = _canonical_name_memory(memory)
        if canonical:
            return canonical
        if has_correction:
            return None
        if not has_correction and re.search(rf"\b(name is|named|called)\s+{re.escape(name)}\b", lower):
            return None
        memory = re.sub(rf"\b{re.escape(name)}\b", "the user", memory, flags=re.IGNORECASE)
        memory = re.sub(r"\bUser,\s+named\s+the user,\s+", "User ", memory, flags=re.IGNORECASE)
        memory = re.sub(r"\bthe user('s)?\b", lambda m: "the user's" if m.group(1) else "the user", memory)
    return " ".join(memory.split())


def _merge_memories(core: list[str], retrieved: list[str], rules: list[dict]) -> list[str]:
    forbidden = _forbidden_names(rules)
    merged: list[str] = []
    seen: set[str] = set()
    for text in [*core, *retrieved]:
        cleaned = _normalise_memory(text, forbidden)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged[:18]


async def get_active_skills(user_id: str) -> list[dict]:
    async with get_conn() as db:
        async with db.execute(
            "SELECT skill_name, trigger, procedure, user_prefs FROM user_skills "
            "WHERE user_id = ? AND active = 1 ORDER BY rowid DESC LIMIT 8",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_active_rules(user_id: str) -> list[dict]:
    async with get_conn() as db:
        async with db.execute(
            "SELECT rule_text, applies_to, confidence FROM user_rules "
            "WHERE user_id = ? AND active = 1 ORDER BY confidence DESC LIMIT 15",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    rules: list[dict] = []
    for row in rows:
        item = dict(row)
        raw_rule = item.get("rule_text", "")
        forbidden = [
            m.group(1).lower()
            for m in re.finditer(r"\bforget\s+([a-z][a-z0-9_-]{1,30})", raw_rule or "", flags=re.IGNORECASE)
        ]
        cleaned = _clean_rule_text(raw_rule)
        if not cleaned:
            continue
        item["rule_text"] = cleaned
        if forbidden:
            item["_forbidden_names"] = forbidden
        rules.append(item)
    return rules


def _build_injection(memories: list[str], skills: list[dict], rules: list[dict]) -> str:
    # Start from the canonical Echo persona — ensures the two code paths
    # (with memory vs. without) always share the same voice and language rules.
    parts = [_get_base_prompt()]

    if memories:
        lines = ["## What you know about this person"]
        for m in memories:
            lines.append(f"- {m}")
        parts.append("\n".join(lines))
    else:
        parts.append(
            "## Memory\n"
            "You do not know much about this person yet. "
            "Listen carefully and ask at most one useful question — "
            "build the picture before making claims about who they are."
        )

    if skills:
        lines = ["## How to help them (apply when the moment fits)"]
        for s in skills:
            line = f"- {s['skill_name']}: {s['procedure']}"
            if s.get("user_prefs"):
                line += f" [{s['user_prefs']}]"
            lines.append(line)
        parts.append("\n".join(lines))

    if rules:
        lines = ["## Rules this person set for you"]
        for r in rules:
            lines.append(f"- {r['rule_text']}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


async def get_context(user_id: str, message: str) -> dict:
    searched, core, skills, rules = await asyncio.gather(
        search_memories(message, user_id=user_id),
        get_core_memories(user_id),
        get_active_skills(user_id),
        get_active_rules(user_id),
    )
    memories = _merge_memories(core, searched, rules)
    return {
        "system_injection": _build_injection(memories, skills, rules),
        "memories": memories,
        "skills": skills,
        "rules": rules,
    }
