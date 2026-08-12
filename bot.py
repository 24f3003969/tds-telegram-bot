import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Minimal HTTP Server for Render Health Check ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def run_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()


threading.Thread(target=run_health_check_server, daemon=True).start()
# --------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
LOG_URL = os.environ.get("LOG_URL", "PASTE_YOUR_PUBLIC_LOG_URL_HERE")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set!")
if not AIPIPE_TOKEN:
    raise ValueError("AIPIPE_TOKEN environment variable is not set!")

client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)
LOG_FILE = "run.jsonl"

# If one model/quota goes bad mid-month, the next one can still answer —
# a single hardcoded model has no fallback if AIPipe access to it changes.
MODELS_TO_TRY = os.environ.get("MODELS_TO_TRY", "gpt-5-mini,gpt-4o-mini,gpt-4.1-mini").split(",")

conversation_history = {}


def log_event(event: dict):
    event["timestamp"] = time.time()
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        logger.exception("Failed to write log event")


SYSTEM_PROMPT = (
    "You are a precise data analyst assistant. The user's message contains a data-analysis question "
    "and specifies an EXACT JSON schema/shape to reply with.\n"
    "CRITICAL RULES:\n"
    "1. Analyze the question carefully and compute the exact answer.\n"
    "2. Output MUST be ONLY a single JSON object matching the requested schema.\n"
    "3. Do NOT include any extra keys such as 'explanation', 'reasoning', 'notes', 'thought', or 'comments'.\n"
    "4. Do NOT wrap output in markdown code fences or conversational text.\n"
    "5. If the request shows a template with 'answer' and 'log_url' keys, output 'answer' containing the "
    "requested answer structure and 'log_url' as a string placeholder."
)


def call_model(history, chat_id):
    last_error = None
    for model_name in MODELS_TO_TRY:
        model_name = model_name.strip()
        if not model_name:
            continue
        for use_json_mode in (True, False):
            try:
                kwargs = dict(
                    model=model_name,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:],
                )
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    log_event({"type": "model_attempt", "chat_id": chat_id, "model": model_name,
                               "json_mode": use_json_mode, "raw_output": content})
                    return content.strip()
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} (json_mode={use_json_mode}) failed: {e}")
                log_event({"type": "model_attempt", "chat_id": chat_id, "model": model_name,
                           "json_mode": use_json_mode, "error": str(e)})
    # Every model/attempt failed — surface the real reason instead of dying silently.
    raise RuntimeError(f"All models failed. Last error: {last_error}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    try:
        reply_text = call_model(history, chat_id)
    except Exception as e:
        logger.exception("call_model failed for all models/attempts")
        log_event({"type": "answer_failed", "chat_id": chat_id, "error": str(e)})
        await update.message.reply_text(json.dumps({"answer": None, "log_url": LOG_URL}))
        return

    history.append({"role": "assistant", "content": reply_text})

    cleaned_text = reply_text
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        start, end = reply_text.find("{"), reply_text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(reply_text[start:end + 1])
            except json.JSONDecodeError:
                parsed = reply_text
        else:
            parsed = reply_text

    if isinstance(parsed, dict):
        for unwanted in ["explanation", "reasoning", "notes", "thought", "comments", "confidence"]:
            parsed.pop(unwanted, None)

        is_log_url_requested = "log_url" in user_text.lower() or "log_url" in parsed
        if is_log_url_requested:
            effective_log_url = (
                LOG_URL if (LOG_URL and LOG_URL != "PASTE_YOUR_PUBLIC_LOG_URL_HERE")
                else "https://raw.githubusercontent.com/username/repo/main/run.jsonl"
            )
            parsed["log_url"] = effective_log_url

    final_reply = json.dumps(parsed) if isinstance(parsed, (dict, list)) else str(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_error_handler(error_handler)
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
logger.info("Bot is running... (Ctrl+C to stop)")
app.run_polling(drop_pending_updates=True)