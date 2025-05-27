
print("backend.py loaded")
from fastapi import FastAPI
import logging

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

        content = response.content
        text = response.text
        status_code = response.status_code
        logging.info(f"HF API response status: {status_code}")
        logging.info(f"HF API response content: {text}")

        try:
            data = response.json()
        except Exception as e:
            logging.error(f"JSON decode error: {e}")
            return {"reply": "Ошибка при обработке ответа от модели."}

    if isinstance(data, list) and "generated_text" in data[0]:
        generated_text = data[0]["generated_text"]
    else:
        generated_text = "Ошибка генерации ответа."

    return {"reply": generated_text}
