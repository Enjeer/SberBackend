from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "EleutherAI/gpt-neo-1.3B"

class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat(message: Message):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    payload = {
        "inputs": message.text,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload
        )
    
    # Сначала проверим статус и выведем ответ для диагностики
    if response.status_code != 200:
        logging.error(f"HuggingFace API returned status {response.status_code}: {response.text}")
        return {"error": f"HuggingFace API returned status {response.status_code}", "detail": response.text}

    try:
        data = response.json()
    except Exception as e:
        logging.error(f"Failed to parse JSON from HuggingFace response: {e}. Raw response: {response.text}")
        return {"error": "Failed to parse JSON from HuggingFace response", "detail": response.text}

    # Возвращаем весь ответ от HF в поле "hf_response"
    return {"hf_response": data}


