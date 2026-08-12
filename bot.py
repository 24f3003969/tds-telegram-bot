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


def get_effective_log_url():
    log_url_env = os.environ.get("LOG_URL")
    if log_url_env and log_url_env != "PASTE_YOUR_PUBLIC_LOG_URL_HERE":
        return log_url_env
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url:
        return render_url.rstrip("/") + "/run.jsonl"
    return "https://raw.githubusercontent.com/username/repo/main/run.jsonl"


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

# Load configuration from environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
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


def parse_json_reply(text: str):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def is_valid_parsed(parsed, is_answer_wrapper_requested):
    if not isinstance(parsed, dict):
        return False
    if is_answer_wrapper_requested:
        ans = parsed.get("answer")
        if ans is None or ans == {} or ans == "":
            return False
    elif not parsed:
        return False
    return True


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
    if len(history) > 10:
        history = history[-10:]
        conversation_history[chat_id] = history

    system_prompt = (
        "You are an expert data analyst AI assistant specializing in Indian public dataset queries (MOSPI, Census 2011, etc.) "
        "and numerical data forecasts.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Compute or retrieve the exact factual/mathematical answer for the user's question.\n"
        "2. Replace ALL schema placeholders (such as '<state name>', '<name>', '<number>', '<value>') with exact factual answers (e.g. 'Assam', 'Delhi', 123.45).\n"
        "3. Output MUST be ONLY a single valid JSON object matching the requested schema.\n"
        "4. If the request template has 'answer' and 'log_url' keys like {\"answer\": ..., \"log_url\": \"...\"}, "
        "output 'answer' containing the computed answer object and 'log_url' as a string placeholder.\n"
        "   Example: {\"answer\": {\"state\": \"Assam\"}, \"log_url\": \"https://example.com/run.jsonl\"}\n"
        "5. NEVER return empty objects like {\"answer\": {}} or empty values. Populate every field with real data.\n"
        "6. Do NOT include any extra keys ('explanation', 'reasoning', 'thought') or markdown formatting."
    )

    is_log_url_requested = "log_url" in user_text.lower()
    is_answer_wrapper_requested = '"answer"' in user_text.lower() or "'answer'" in user_text.lower()

    models_to_try = ["gpt-5-mini", "gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"]
    parsed = None
    reply_text = ""

    for model_name in models_to_try:
        # Attempt 1: with response_format json_object
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}] + history[-6:],
                response_format={"type": "json_object"}
            )
            candidate_text = response.choices[0].message.content.strip()
            candidate_parsed = parse_json_reply(candidate_text)
            if is_valid_parsed(candidate_parsed, is_answer_wrapper_requested):
                reply_text = candidate_text
                parsed = candidate_parsed
                break
        except Exception as e:
            logger.warning(f"Failed JSON object API call with model {model_name}: {e}")

        # Attempt 2: standard completion
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}] + history[-6:]
            )
            candidate_text = response.choices[0].message.content.strip()
            candidate_parsed = parse_json_reply(candidate_text)
            if is_valid_parsed(candidate_parsed, is_answer_wrapper_requested):
                reply_text = candidate_text
                parsed = candidate_parsed
                break
        except Exception as e:
            logger.warning(f"Failed standard API call with model {model_name}: {e}")

    # Fallback retry if model returned empty answer structure
    if not is_valid_parsed(parsed, is_answer_wrapper_requested):
        logger.warning("Model returned empty or invalid answer structure. Running fallback extraction...")
        fallback_prompt = (
            f"Question: {user_text}\n\n"
            "Answer the question accurately. Replace placeholders like '<state name>' or '<name>' with real data. "
            "Reply with ONLY a valid JSON object."
        )
        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": fallback_prompt}]
                )
                candidate_text = response.choices[0].message.content.strip()
                candidate_parsed = parse_json_reply(candidate_text)
                if candidate_parsed and candidate_parsed != {}:
                    parsed = candidate_parsed
                    reply_text = candidate_text
                    break
            except Exception:
                continue

    if parsed is None:
        parsed = {}

    history.append({"role": "assistant", "content": reply_text or json.dumps(parsed)})

    effective_log_url = get_effective_log_url()

    if isinstance(parsed, dict):
        for unwanted in ["explanation", "reasoning", "notes", "thought", "comments", "confidence"]:
            parsed.pop(unwanted, None)

        if is_log_url_requested and is_answer_wrapper_requested:
            ans = parsed.get("answer")
            if ans is None or ans == {}:
                # LLM outputted answer keys directly at root level
                answer_content = {k: v for k, v in parsed.items() if k not in ("log_url", "answer")}
                parsed = {
                    "answer": answer_content,
                    "log_url": effective_log_url
                }
            else:
                parsed["log_url"] = effective_log_url
        elif is_log_url_requested:
            parsed["log_url"] = effective_log_url
        else:
            parsed.pop("log_url", None)
    elif is_log_url_requested and is_answer_wrapper_requested:
        parsed = {
            "answer": parsed,
            "log_url": effective_log_url
        }

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

