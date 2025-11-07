import os
import logging
from fastapi import FastAPI, Request, Response
from telegram import Update, Bot
from main import bot_application # Импортируем Application из main.py

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Получение токена из переменной окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не установлен. Приложение не может быть запущено.")
    # Выход или поднятие исключения, но для FastAPI просто продолжим
    
# Инициализация FastAPI
app = FastAPI(
    title="QuantumTrader Telegram Bot",
    description="Telegram bot for crypto trading analysis and recommendations.",
    version="1.0.0",
)

# Инициализация Bot
bot = Bot(TOKEN)

@app.on_event("startup")
async def startup_event():
    """Действия при запуске приложения."""
    # В среде Render WebHook должен быть установлен вручную или через CI/CD.
    # Мы не будем устанавливать WebHook здесь, так как Render сам предоставляет URL.
    # Telegram будет отправлять обновления на /webhook.
    logger.info("FastAPI application started.")

@app.post("/webhook")
async def webhook_handler(request: Request):
    """Обрабатывает входящие обновления от Telegram."""
    try:
        # Получаем JSON-тело запроса
        data = await request.json()
        
        # Создаем объект Update из данных
        update = Update.de_json(data, bot)
        
        # Обрабатываем обновление с помощью Application
        await bot_application.update_queue.put(update)
        
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Ошибка обработки WebHook: {e}")
        return Response(status_code=500)

@app.get("/")
async def root():
    """Простая проверка работоспособности."""
    return {"message": "QuantumTrader Bot is running. Send updates to /webhook"}

# Для запуска uvicorn main:app
# В Render будет использоваться команда: uvicorn app:app --host 0.0.0.0 --port $PORT
