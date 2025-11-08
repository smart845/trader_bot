import os
import logging
from fastapi import FastAPI, Request, Response
from telegram import Update, Bot
from main import bot_application  # берем объект Application из main.py

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Получаем токен из переменных окружения Render
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    logger.warning("⚠️ TELEGRAM_BOT_TOKEN не установлен в переменных окружения Render!")

# Создаём FastAPI-приложение
app = FastAPI()

@app.get("/")
async def home():
    """Главная страница — проверка, что бот жив"""
    return {"message": "QuantumTrader Bot is running. Send updates to /webhook"}

@app.get("/webhook")
async def webhook_get():
    """Просто для ручной проверки в браузере"""
    return {"message": "WebHook is active. Send POST requests with Telegram updates."}

@app.post("/webhook")
async def webhook_post(request: Request):
    """Основной обработчик Telegram Webhook"""
    try:
        data = await request.json()
        bot = Bot(token=TOKEN)
        update = Update.de_json(data, bot)
        await bot_application.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.exception(f"Ошибка при обработке webhook: {e}")
        return Response(status_code=500)
