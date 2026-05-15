import asyncio
import httpx
import json

async def seed():
    url = "http://localhost:8002/save"
    user_id = "default_user" # We'll see what the app uses, but seeding a default helps
    
    memories = [
        ("My name is ASUS. I am a developer working on AI integrations.", "Nice to meet you ASUS! I'll remember that."),
        ("I am building a voice tutoring agent called SAGE using Deepgram and Gemini.", "SAGE sounds like an ambitious project. I've noted the tech stack."),
        ("We are currently integrating ECHO as a sidecar for ChatMCP.", "Understood. The sidecar architecture for ChatMCP is the current focus."),
    ]
    
    async with httpx.AsyncClient() as client:
        for u, a in memories:
            resp = await client.post(url, json={
                "user_id": user_id,
                "user_message": u,
                "assistant_message": a,
                "model_used": "manual-seed",
                "engagement_signal": "seed"
            })
            print(f"Seeded: {u[:30]}... Status: {resp.status_code}")

if __name__ == "__main__":
    asyncio.run(seed())
