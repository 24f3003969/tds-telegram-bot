import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# Set up logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

LOG_FILE = "run.jsonl"

# --- HTTP Server for Render Health Check & Public Logs ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/run.jsonl", "/logs"):
            if os.path.exists(LOG_FILE):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-jsonlines")
                self.end_headers()
                with open(LOG_FILE, "rb") as f:
                    self.wfile.write(f.read())
                return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress routine HTTP health check log noise
        pass


def run_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    logger.info(f"Health check server listening on port {port}")
    server.serve_forever()


# Start health check server in background thread to pass Render Web Service checks
threading.Thread(target=run_health_check_server, daemon=True).start()
# --------------------------------------------------

# Load configuration from environment variables (do NOT hardcode secrets)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
LOG_URL = os.environ.get("LOG_URL", "PASTE_YOUR_PUBLIC_LOG_URL_HERE")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set!")
if not AIPIPE_TOKEN:
    raise ValueError("AIPIPE_TOKEN or OPENAI_API_KEY environment variable is not set!")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=AIPIPE_TOKEN)

# Keeps last few messages per chat for multi-turn questions
conversation_history = {}


def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = "Hello! I am a data analysis Telegram bot. Send me your query or JSON request."
    await update.message.reply_text(welcome_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    # Keep history bounded to avoid excessive memory / token growth
    if len(history) > 10:
        history = history[-10:]
        conversation_history[chat_id] = history

    system_prompt = (
        "You are a precise data analyst assistant. The user's message contains a data-analysis question "
        "and specifies an EXACT JSON schema/shape to reply with.\n"
        "CRITICAL RULES:\n"
        "1. Analyze the question carefully and compute the exact answer.\n"
        "2. Output MUST be ONLY a single JSON object matching the requested schema.\n"
        "3. Do NOT include any extra keys such as 'explanation', 'reasoning', 'notes', 'thought', or 'comments'.\n"
        "4. Do NOT wrap output in markdown code fences or conversational text.\n"
        "5. If the request shows a template with 'answer' and 'log_url' keys, output 'answer' containing the requested answer structure and 'log_url' as a string placeholder."
    )

    reply_text = ""
    models_to_try = ["gpt-5-mini", "gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"]
    
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}] + history[-6:],
                response_format={"type": "json_object"}
            )
            reply_text = response.choices[0].message.content.strip()
            break
        except Exception:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}] + history[-6:]
                )
                reply_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                logger.warning(f"Failed API call with model {model_name}: {e}")
                continue

    if not reply_text:
        reply_text = "{}"

    history.append({"role": "assistant", "content": reply_text})

    # Clean up markdown code blocks if present
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

    # Post-processing: remove unwanted extra keys that LLM might hallucinate
    if isinstance(parsed, dict):
        for unwanted in ["explanation", "reasoning", "notes", "thought", "comments", "confidence"]:
            parsed.pop(unwanted, None)

        # Handle log_url strictly: inject log_url ONLY IF requested in user_text
        is_log_url_requested = "log_url" in user_text.lower()
        if is_log_url_requested:
            effective_log_url = LOG_URL if (LOG_URL and LOG_URL != "PASTE_YOUR_PUBLIC_LOG_URL_HERE") else "https://raw.githubusercontent.com/username/repo/main/run.jsonl"
            parsed["log_url"] = effective_log_url
        else:
            parsed.pop("log_url", None)

    final_reply = json.dumps(parsed) if isinstance(parsed, (dict, list)) else str(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting... Listening for Telegram updates.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

