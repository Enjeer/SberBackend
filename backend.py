from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# Разрешаем запросы с твоего фронта на localhost:3000 и с фронта в проде (замени на свой фронт URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Установи в Render environment variables
HF_MODEL = "tiiuae/falcon-7b-instruct"  # Обученная Falcon 7B модель с HuggingFace Hub

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
            "max_new_tokens": 100,
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

    data = response.json()
    # Обработка формата ответа HuggingFace Inference API для текстовых моделей
    if isinstance(data, list) and "generated_text" in data[0]:
        generated_text = data[0]["generated_text"]
    else:
        generated_text = "Ошибка генерации ответа."

    return {"reply": generated_text}
