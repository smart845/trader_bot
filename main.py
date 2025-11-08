import os
import logging
import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import warnings
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Класс для анализа криптовалют (из предоставленного кода) ---

class AdvancedCryptoAnalyzer:
    def __init__(self):
        warnings.filterwarnings('ignore')
        
    def get_binance_data(self, symbol: str, interval: str = '1h', limit: int = 500):
        """Получение данных с Binance"""
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        data = requests.get(url, params=params).json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Конвертация типов
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
            
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df

    def get_funding_rate(self, symbol: str):
        """Получение фандинг рейта"""
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {'symbol': symbol}
            data = requests.get(url, params=params).json()
            # Находим нужный символ в списке, если это список
            if isinstance(data, list):
                for item in data:
                    if item.get('symbol') == symbol:
                        return float(item.get('lastFundingRate', 0))
                return 0
            # Если это один объект
            return float(data.get('lastFundingRate', 0))
        except Exception as e:
            logger.error(f"Ошибка при получении фандинг рейта для {symbol}: {e}")
            return 0

    def get_open_interest(self, symbol: str):
        """Получение открытого интереса"""
        try:
            url = "https://fapi.binance.com/fapi/v1/openInterest"
            params = {'symbol': symbol}
            data = requests.get(url, params=params).json()
            return float(data.get('openInterest', 0))
        except Exception as e:
            logger.error(f"Ошибка при получении открытого интереса для {symbol}: {e}")
            return 0

    def calculate_all_indicators(self, df):
        """Расчет всех возможных индикаторов (сокращенная версия для скорости)"""
        df = df.copy()
        
        # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # === МОМЕНТУМ ИНДИКАТОРЫ ===
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # === ВОЛАТИЛЬНОСТЬ ===
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # === КАСТОМНЫЕ РАСЧЕТЫ ===
        df['taker_buy_volume'] = pd.to_numeric(df['taker_buy_quote'])
        df['quote_volume'] = pd.to_numeric(df['quote_volume'])
        df['taker_sell_volume'] = df['quote_volume'] - df['taker_buy_volume']
        df['volume_delta'] = df['taker_buy_volume'] - df['taker_sell_volume']
        df['volume_delta_ratio'] = df['volume_delta'] / df['quote_volume']
        
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        
        return df.dropna()

    def calculate_market_structure(self, df):
        """Анализ рыночной структуры"""
        if df.empty:
            return None
            
        latest = df.iloc[-1]
        
        # Определение тренда
        trend_short = "BULL" if latest['close'] > latest['ema_20'] else "BEAR"
        trend_medium = "BULL" if latest['close'] > latest['ema_50'] else "BEAR"
        trend_long = "BULL" if latest['close'] > latest['ema_200'] else "BEAR"
        
        # Уровни перекупленности/перепроданности
        overbought = latest['rsi_14'] > 70
        oversold = latest['rsi_14'] < 30
        
        # Сигналы индикаторов
        signals = {
            'macd_bullish': latest['macd'] > latest['macd_signal'],
            'rsi_bullish': latest['rsi_14'] > 50,
            'stoch_bullish': latest['stoch_k'] > latest['stoch_d'],
        }
        
        bull_signals = sum(signals.values())
        total_signals = len(signals)
        
        return {
            'trend_short': trend_short,
            'trend_medium': trend_medium,
            'trend_long': trend_long,
            'overbought': overbought,
            'oversold': oversold,
            'bullish_score': bull_signals / total_signals,
            'signals': signals
        }

    def calculate_entry_points(self, df, market_structure):
        """Расчет точек входа и выхода"""
        if df.empty or market_structure is None:
            return {'direction': 'HOLD', 'confidence': 0}
            
        latest = df.iloc[-1]
        current_price = latest['close']
        atr = latest['atr']
        atr_multiplier = 2
        
        direction = 'HOLD'
        entry, stop_loss, take_profit_1, take_profit_2, take_profit_3 = None, None, None, None, None
        
        if market_structure['bullish_score'] >= 0.6 and not market_structure['overbought']:
            # LONG сигнал
            entry = current_price * 0.999  # Немного ниже текущей цены
            stop_loss = latest['support']
            take_profit_1 = entry + atr * 1.5
            take_profit_2 = entry + atr * 3
            take_profit_3 = latest['resistance']
            direction = "LONG"
            
        elif market_structure['bullish_score'] <= 0.4 and not market_structure['oversold']:
            # SHORT сигнал
            entry = current_price * 1.001  # Немного выше текущей цены
            stop_loss = latest['resistance']
            take_profit_1 = entry - atr * 1.5
            take_profit_2 = entry - atr * 3
            take_profit_3 = latest['support']
            direction = "SHORT"
        else:
            return {
                'direction': 'HOLD',
                'entry': None,
                'stop_loss': None,
                'take_profits': [],
                'confidence': market_structure['bullish_score']
            }
        
        # Расчет риска и доходности
        risk = abs(entry - stop_loss)
        
        take_profits = [take_profit_1, take_profit_2, take_profit_3]
        
        tp_results = []
        for tp in take_profits:
            reward = abs(tp - entry)
            risk_reward = reward / risk if risk > 0 else 0
            tp_results.append({'level': round(tp, 4), 'rr_ratio': round(risk_reward, 2)})
        
        risk_per_trade = f"{round(risk/current_price*100, 2)}%"
        
        return {
            'direction': direction,
            'entry': round(entry, 4),
            'stop_loss': round(stop_loss, 4),
            'take_profits': tp_results,
            'risk_per_trade': risk_per_trade,
            'confidence': round(market_structure['bullish_score'], 3)
        }

    def analyze_coin(self, coin: str):
        """Полный анализ монеты"""
        symbol = f"{coin.upper()}USDT"
        
        try:
            # Получение данных
            df = self.get_binance_data(symbol)
            if df.empty:
                return {"error": f"Не удалось получить данные для {symbol}. Проверьте правильность символа."}
            
            # Расчет индикаторов
            df = self.calculate_all_indicators(df)
            
            # Дополнительные метрики
            funding_rate = self.get_funding_rate(symbol)
            open_interest = self.get_open_interest(symbol)
            
            # Анализ структуры
            market_structure = self.calculate_market_structure(df)
            
            # Торговые уровни
            trading_levels = self.calculate_entry_points(df, market_structure)
            
            # Сбор всех метрик
            latest = df.iloc[-1]
            
            result = {
                'coin': coin.upper(),
                'current_price': round(latest['close'], 4),
                'price_change_24h': round((latest['close'] - df.iloc[-24]['close']) / df.iloc[-24]['close'] * 100, 2),
                'market_metrics': {
                    'funding_rate': round(funding_rate * 100, 4),
                    'open_interest': open_interest,
                    'volume_delta': round(latest['volume_delta_ratio'] * 100, 2),
                },
                'market_structure': market_structure,
                'trading_recommendation': trading_levels,
                'support_resistance': {
                    'support': round(latest['support'], 4),
                    'resistance': round(latest['resistance'], 4),
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Критическая ошибка анализа для {symbol}: {e}")
            return {"error": f"Критическая ошибка анализа: {str(e)}"}

# --- Функции Telegram-бота ---

analyzer = AdvancedCryptoAnalyzer()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /start и выводит приветствие."""
    user_name = update.message.from_user.first_name
    
    welcome_message = (
        f"👋 Привет, *{user_name}*! Я *QuantumTrader* — твой личный помощник в мире криптотрейдинга. 🤖\n\n"
        "Я анализирую рынок, используя продвинутые технические индикаторы и метрики, чтобы дать тебе *реально работающие* торговые рекомендации.\n\n"
        "✨ *Что я умею:*\n"
        "1. 📊 Анализировать любую монету с привязкой к USDT (например, BTC, ETH, SOL).\n"
        "2. 📈 Предоставлять торговые рекомендации (LONG/SHORT/HOLD) с точками входа, стоп-лоссами и тейк-профитами.\n"
        "3. 💡 Оценивать рыночную структуру и настроение (бычий/медвежий тренд).\n\n"
        "👇 *Список команд:*\n"
        "/start - Показать это приветствие и список команд.\n"
        "/analyze `<COIN>` - Получить торговый прогноз по монете (например, `/analyze BTC`).\n"
        "/help - Показать краткую справку."
    )
    
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /help."""
    help_message = (
        "💡 *Справка по QuantumTrader*\n\n"
        "Для получения прогноза используй команду:\n"
        "`/analyze <COIN>`\n\n"
        "Пример: `/analyze ETH`\n\n"
        "Я проанализирую данные с Binance и выдам подробный отчет с рекомендацией."
    )
    await update.message.reply_text(help_message, parse_mode='Markdown')

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /analyze <COIN>."""
    if not context.args:
        await update.message.reply_text(
            "❌ *Ошибка:* Укажи тикер монеты. Пример: `/analyze BTC`",
            parse_mode='Markdown'
        )
        return

    coin = context.args[0].upper()
    await update.message.reply_text(f"⏳ Анализирую *{coin}USDT*... Это может занять до 10 секунд.", parse_mode='Markdown')

    result = analyzer.analyze_coin(coin)

    if "error" in result:
        await update.message.reply_text(f"❌ *Ошибка анализа:* {result['error']}", parse_mode='Markdown')
        return

    # Форматирование результата
    rec = result['trading_recommendation']
    structure = result['market_structure']
    levels = result['support_resistance']
    metrics = result['market_metrics']
    
    # Эмодзи для направления
    direction_emoji = "🟢 LONG" if rec['direction'] == "LONG" else "🔴 SHORT" if rec['direction'] == "SHORT" else "🟡 HOLD"
    
    # Формирование сообщения
    report_message = (
        f"📈 *ОТЧЕТ QUANTUMTRADER: {result['coin']}USDT* 📉\n"
        f"_{result['timestamp']} (1H таймфрейм)_\n\n"
        
        f"💰 *ТЕКУЩАЯ ЦЕНА:* `${result['current_price']}`\n"
        f"Изменение за 24ч: `{result['price_change_24h']}%`\n\n"
        
        f"🎯 *РЕКОМЕНДАЦИЯ:* {direction_emoji}\n"
        f"Уверенность: `{round(rec['confidence'] * 100)}%`\n"
    )
    
    if rec['direction'] != 'HOLD':
        tp_list = "\n".join([f"  - TP{i+1}: `${tp['level']}` (R:R `{tp['rr_ratio']}`)" for i, tp in enumerate(rec['take_profits'])])
        report_message += (
            f"\n"
            f"➡️ *ТОЧКА ВХОДА:* `${rec['entry']}`\n"
            f"🛑 *СТОП-ЛОСС:* `${rec['stop_loss']}` (`{rec['risk_per_trade']}`)\n"
            f"✅ *ТЕЙК-ПРОФИТЫ:*\n{tp_list}\n"
        )
    
    report_message += (
        f"\n"
        f"📊 *РЫНОЧНАЯ СТРУКТУРА:*\n"
        f"Тренд (К/С/Д): `{structure['trend_short']}/{structure['trend_medium']}/{structure['trend_long']}`\n"
        f"RSI 14: `{result['market_structure']['oversold'] or result['market_structure']['overbought']}`\n"
        f"Поддержка/Сопротивление: `${levels['support']}` / `${levels['resistance']}`\n\n"
        
        f"💡 *МЕТРИКИ:*\n"
        f"Фандинг рейт: `{metrics['funding_rate']}%`\n"
        f"Дельта объема: `{metrics['volume_delta']}%`\n"
        f"Открытый интерес: `{metrics['open_interest']:,.0f}`\n"
    )
    
    await update.message.reply_text(report_message, parse_mode='Markdown')

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает неизвестные команды."""
    await update.message.reply_text(
        "🤔 Неизвестная команда. Используй /start для списка команд.",
        parse_mode='Markdown'
    )

def main() -> None:
    """Запуск бота."""
    # Получение токена из переменной окружения
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен. Бот не может быть запущен.")
        return

    # Создание Application и передача токена
    application = Application.builder().token(TOKEN).build()

    # Обработчики команд
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_command))

    # Обработчик неизвестных команд (должен быть последним)
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # Запуск бота (для локального тестирования)
    # application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    # Для деплоя на Render (WebHook)
    # Render будет использовать uvicorn для запуска этого файла как ASGI-приложения
    # Мы будем использовать FastAPI для создания WebHook-сервера
    
    # Для простоты деплоя на Render, мы будем использовать long polling, 
    # так как настройка WebHook требует дополнительного кода с FastAPI/Flask, 
    # что усложнит простой деплой. Render поддерживает постоянные процессы.
    
    # application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    # ВНИМАНИЕ: Для деплоя на Render, который требует WebHook, необходимо использовать 
    # библиотеку `telegram.ext.ExtBot` и настроить FastAPI/Flask.
    # Для упрощения, я создам минимальный WebHook-сервер на FastAPI, как это часто делается.
    
    # --- WebHook Setup (для Render) ---
    from fastapi import FastAPI
    from telegram import Bot, Update
    
    app = FastAPI()
    bot = Bot(TOKEN)
    
    @app.post("/webhook")
    async def webhook_handler(request: dict):
        """Обрабатывает входящие обновления от Telegram."""
        update = Update.de_json(request, bot)
        await application.process_update(update)
        return {"message": "ok"}

    # Запуск application.run_polling() для локального тестирования
    # Для деплоя на Render, uvicorn запустит app, и WebHook будет настроен вручную
    # через API Telegram.
    
    # В Render мы будем использовать команду: uvicorn main:app --host 0.0.0.0 --port $PORT
    # Для этого нам нужно, чтобы application был доступен для WebHook.
    
    # Настройка WebHook в main()
    async def post_init(application: Application):
        """Настройка WebHook после инициализации."""
        # Установка WebHook должна происходить вне цикла run_polling
        # Но для простоты деплоя на Render, где нет публичного IP для установки WebHook,
        # мы оставим WebHook-логику в FastAPI и будем использовать application.process_update
        # Render сам предоставит публичный URL.
        pass
        
    application.post_init = post_init
    
    # В Render, uvicorn запустит FastAPI-приложение 'app'.
    # Мы должны убедиться, что application.process_update() работает корректно.
    
    # Для деплоя на Render, который поддерживает постоянные процессы, 
    # *самый простой* способ - использовать `run_polling` в отдельном потоке, 
    # но это несовместимо с FastAPI.
    
    # *Правильный* способ для Render:
    # 1. Запустить uvicorn main:app
    # 2. Настроить WebHook на адрес Render.
    
    # Для упрощения, я буду использовать `run_polling` для локального запуска, 
    # и предоставлю инструкцию для WebHook на Render.
    
    # Для деплоя на Render, мы будем использовать WebHook.
    # Создадим отдельный файл `app.py` для FastAPI, чтобы разделить логику.
    
    # Для текущего файла `main.py` я оставлю только логику бота, 
    # а WebHook-обвязку сделаю в отдельном файле `app.py`.
    
    # Инициализация Application
    application = Application.builder().token(TOKEN).build()
    
    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Сохраняем application в глобальной области видимости для app.py
    global bot_application
    bot_application = application

# Запуск main для инициализации application
main()

# --- WebHook Server (для app.py) ---
# Создадим минимальный WebHook-сервер, который будет использовать `bot_application`
# Это будет в отдельном файле `app.py` для чистоты.

# ВАЖНО: Для деплоя на Render, нам нужно использовать WebHook.
# Я создам файл `app.py` и `Procfile`.

# Конец main.py
