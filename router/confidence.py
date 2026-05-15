from db.database import get_conn

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "coding":   ["python", "javascript", "typescript", "code", "function", "bug", "error",
                 "import", "class", "async", "api", "sql", "git", "docker", "npm", "pip",
                 "dart", "flutter", "endpoint", "backend", "server", "database", "deploy",
                 "fastapi", "django", "nodejs", "react", "css", "html", "http", "rest",
                 "graphql", "redis", "postgres", "sqlite", "bash", "shell", "linux"],
    "ml":       ["model", "training", "dataset", "epoch", "loss", "lora", "finetune",
                 "embedding", "inference", "tensor", "gradient", "llm", "ai", "vllm",
                 "qlora", "unsloth", "llamafactory", "transformer", "attention", "token",
                 "prompt", "fine-tuning", "rag", "vector", "openai", "claude", "gemini",
                 "shadow clone", "adapter", "checkpoint", "quantization"],
    "writing":  ["write", "essay", "draft", "edit", "rewrite", "grammar", "blog",
                 "article", "summarize", "translate", "email", "letter", "report",
                 "description", "caption", "content", "copy", "pitch"],
    "math":     ["calculate", "equation", "formula", "solve", "derivative", "integral",
                 "probability", "statistics", "percentage", "ratio", "average", "median"],
    "research": ["research", "paper", "study", "explain", "what is", "how does",
                 "compare", "difference between", "pros and cons", "overview", "history",
                 "trend", "industry", "market", "technology"],
    "personal": ["feel", "feeling", "emotion", "anxious", "anxiety", "stress", "lonely",
                 "lonely", "social", "friend", "relationship", "family", "life", "goal",
                 "habit", "routine", "morning", "sleep", "health", "workout", "balance",
                 "motivation", "confidence", "fear", "worry", "happy", "sad", "tired",
                 "painting", "hobby", "travel", "vacation", "book", "movie", "music",
                 "alone", "introvert", "people", "conversation", "communicate", "skill"],
    "work":     ["work", "job", "career", "salary", "interview", "meeting", "project",
                 "client", "deadline", "manager", "team", "office", "startup", "business",
                 "freelance", "revenue", "dialogo", "product", "feature", "launch",
                 "strategy", "planning", "roadmap", "sprint", "agile"],
    "language": ["spanish", "french", "german", "italian", "portuguese", "japanese",
                 "chinese", "korean", "translate", "grammar", "vocabulary", "learn.*language",
                 "practice.*language", "speak", "fluent", "native", "accent", "pronunciation",
                 "english", "bilingual"],
    "general":  [],
}

# Priority order — first match wins
_TOPIC_ORDER = ["coding", "ml", "writing", "math", "research", "personal", "work", "language"]


def detect_topic(message: str) -> str:
    msg_lower = message.lower()
    for topic in _TOPIC_ORDER:
        if any(kw in msg_lower for kw in TOPIC_KEYWORDS[topic]):
            return topic
    return "general"


async def get_confidence(user_id: str, message: str) -> float:
    topic = detect_topic(message)
    async with get_conn() as db:
        async with db.execute(
            "SELECT topic, score FROM confidence WHERE user_id = ? AND topic IN (?, 'general')",
            (user_id, topic),
        ) as cur:
            rows = await cur.fetchall()
    scores = {r["topic"]: float(r["score"]) for r in rows}
    # Use the higher of topic-specific or general confidence as the floor
    return max(scores.get(topic, 0.0), scores.get("general", 0.0))


async def update_confidence(user_id: str, topic: str, model_used: str) -> None:
    # Gemma handled it → boost (model is serving the user)
    # Teacher handled it → small boost but apply decay to general
    # Decay prevents permanent lock-in at 1.0 so teacher still runs occasionally
    if model_used == "local":
        delta = 0.03
        decay = 0.0
    else:
        delta = 0.01
        decay = 0.005  # slow general decay when teacher is used

    async with get_conn() as db:
        # Update specific topic score
        await db.execute(
            """
            INSERT INTO confidence (user_id, topic, score)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, topic) DO UPDATE
            SET score      = MAX(0.0, MIN(1.0, score + ?)),
                updated_at = datetime('now')
            """,
            (user_id, topic, max(0.0, delta), delta),
        )
        # Apply gentle decay to general score when teacher handles a request
        if decay > 0 and topic != "general":
            await db.execute(
                """
                UPDATE confidence SET score = MAX(0.5, score - ?)
                WHERE user_id = ? AND topic = 'general'
                """,
                (decay, user_id),
            )
        await db.commit()
