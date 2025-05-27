# backend.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Поставь туда свой токен через env var или прямо в коде (не рекомендуется)
HF_MODEL = "gpt2"  # Замени на нужную модель

class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat(message: Message):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    payload = {
        "inputs": message.text
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload
        )
    data = response.json()
    # Пример для текстового ответа модели
    generated_text = data[0]["generated_text"] if isinstance(data, list) else "Error"

    return {"reply": generated_text}
