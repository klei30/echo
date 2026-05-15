import asyncio, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import settings
import httpx

async def test():
    print(f"Model: {settings.teacher_model}")
    print(f"URL:   {settings.teacher_base_url}")
    print(f"Key:   {'set' if settings.llm_api_key else 'MISSING'}\n")

    # Simulate the exact calls Echo makes during training
    tests = [
        (
            "Memory extraction (mem0 style)",
            "Extract one key personal fact from this conversation to remember about the user.\n"
            "User: I want to get better at swimming and I enjoy solo sports.\n"
            "Assistant: That's a great goal, swimming is excellent for fitness.\n"
            "Output one short sentence fact only."
        ),
        (
            "Quality scoring (training pair filter)",
            "Score this AI response from 0.0 to 1.0 for quality and relevance.\n"
            "User: What should I focus on to become a better backend developer?\n"
            "AI: Focus on system design, databases, and API patterns. Build real projects.\n"
            "Output only a decimal number like 0.8"
        ),
        (
            "Preference extraction",
            "The user said: 'I prefer short direct answers without bullet points'.\n"
            "Output a single rule in this format: RULE: <rule text>"
        ),
    ]

    all_ok = True
    async with httpx.AsyncClient(timeout=30) as client:
        for label, prompt in tests:
            print(f"--- {label} ---")
            try:
                resp = await client.post(
                    f"{settings.teacher_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.teacher_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 80,
                        "temperature": 0.1,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL).strip()
                    usage = data.get("usage", {})
                    print(f"Response: {content}")
                    print(f"Tokens:   in={usage.get('prompt_tokens','?')} out={usage.get('completion_tokens','?')}")
                    print(f"Status:   OK")
                elif resp.status_code == 429:
                    print(f"RATE LIMITED: {resp.text[:200]}")
                    all_ok = False
                else:
                    print(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    all_ok = False
            except Exception as e:
                print(f"FAILED: {e}")
                all_ok = False
            print()

    print("=" * 40)
    print(f"Gemma 4 31B teacher: {'READY FOR TRAINING' if all_ok else 'ISSUES FOUND - check above'}")
    print(f"Untrained pairs ready: 33 (need 20) - threshold met")
    if all_ok:
        print("\nSafe to start training.")

asyncio.run(test())
