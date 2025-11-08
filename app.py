import os
import logging
from fastapi import FastAPI, Request, Response
from telegram import Update, Bot
from main import bot_application  # Application instance created in main.py

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Token used only to build Update objects (python-telegram-bot requires a Bot instance here)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN is not set. Updates may fail to deserialize.")

app = FastAPI()

@app.get("/webhook")
async def webhook_get():
    """Health-check endpoint. Telegram uses POST; GET is for manual checks."""
    return {"message": "WebHook is active. Send POST requests with Telegram updates."}

@app.post('/webhook')
async def webhook_post(request: Request):
    """Receive Telegram updates and pass them to python-telegram-bot Application."""
    try:
        data = await request.json()
        # Build Update using a Bot instance
        bot = Bot(token=TOKEN) if TOKEN else None
        update = Update.de_json(data, bot)
        if 'bot_application' not in globals() or bot_application is None:
            logger.error("bot_application is not initialized.")
            return Response(status_code=500)
        await bot_application.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.exception("Failed to process webhook: %s", e)
        return Response(status_code=500)

@app.get("/")
async def root():
    return {"message": "QuantumTrader Bot is running. Send updates to /webhook"}
