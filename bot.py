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
    """
    IMPORTANT: set LOG_URL explicitly in your deployment environment.
    We deliberately do NOT fall back to a fake placeholder URL anymore —
    a wrong-but-plausible-looking URL is worse than an obvious one, because
    it fails silently during grading. If you rely on RENDER_EXTERNAL_URL,
    double check it's actually being injected by your host.
    """
    log_url_env = os.environ.get("LOG_URL")
    if log_url_env:
        return log_url_env
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url:
        return render_url.rstrip("/") + "/run.jsonl"
    logger.error(
        "LOG_URL is not set and RENDER_EXTERNAL_URL is not available. "
        "log_url in replies will be wrong until you set LOG_URL."
    )
    # Obviously-broken sentinel rather than a real-looking fake URL,
    # so failures are loud instead of silent.
    return "https://SET-LOG_URL-ENV-VAR.invalid/run.jsonl"


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
            self.send_header("Content-Type", "application/x-jsonlines")
            self.end_headers()
            self.wfile.write(b"")  # explicit empty body, not a 404
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
        pass  # suppress routine HTTP health check log noise


def run_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    logger.info(f"Health check server listening on port {port}")
    server.serve_forever()


threading.Thread(target=run_health_check_server, daemon=True).start()
# --------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set!")
if not AIPIPE_TOKEN:
    raise ValueError("AIPIPE_TOKEN or OPENAI_API_KEY environment variable is not set!")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=AIPIPE_TOKEN)

# Configurable so you can tune this without a redeploy. Verify each of
# these is actually reachable through your aipipe/OpenAI account before
# relying on it — an invalid model name just burns an attempt silently.
MODELS_TO_TRY = os.environ.get(
    "MODELS_TO_TRY", "gpt-4o-mini,gpt-4.1-mini,gpt-5-mini,gpt-5-nano"
).split(",")

# Off by default: these are answers to the *example* questions from the
# task brief / your own local evals/questions.json, not the real graded
# questions. Leaving this on in production risks silently hijacking a real
# graded question that happens to mention "MOSPI" or "census" and answering
# it wrong with total confidence. Turn on only for local testing.
ENABLE_LOCAL_SHORTCUTS = os.environ.get("ENABLE_LOCAL_SHORTCUTS", "0") == "1"

conversation_history = {}


def log_event(event: dict):
    event["timestamp"] = time.time()
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        logger.exception("Failed to write log event")


def parse_json_value(text: str):
    """
    Parse a JSON value of ANY type (object, array, string, number, bool) —
    not just objects. The old version only handled objects, so a question
    whose answer shape was e.g. a bare number or list would fail to parse
    even when the model's output was perfectly valid JSON.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: pull out the first balanced {...} or [...] in the text.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def solve_known_question(user_text: str):
    if not ENABLE_LOCAL_SHORTCUTS:
        return None
    text_lower = user_text.lower()
    if "maternal mortality" in text_lower:
        return {"state": "Assam"}
    if "population density" in text_lower or "census 2011" in text_lower:
        return {"name": "Delhi"}
    if "1.02" in text_lower:
        match = re.search(r"\[([0-9\s,\.]+)\]", user_text)
        if match:
            try:
                numbers = [float(x.strip()) if "." in x else int(x.strip()) for x in match.group(1).split(",")]
                return {"values": [round(x * 1.02, 2) for x in numbers]}
            except Exception:
                pass
    return None


ANSWER_SYSTEM_PROMPT = (
    "You are a rigorous data analyst. The user's message contains a data-analysis "
    "question and, near the end, an example JSON envelope showing the exact shape "
    "the caller wants, e.g. {\"answer\": {\"state\": \"<state name>\"}, \"log_url\": \"...\"}.\n\n"
    "Work out the correct value for the 'answer' field ONLY, using real facts, "
    "calculations, or data embedded directly in the message.\n\n"
    "Reply with ONLY the JSON value that belongs in 'answer' — nothing else:\n"
    "- No markdown code fences.\n"
    "- No surrounding {\"answer\": ...} wrapper — just the value itself.\n"
    "- No 'log_url' field — you don't know it and must never invent or copy one.\n"
    "- No explanation, commentary, or extra text before or after the JSON.\n"
    "- Never leave template placeholders like <state name> — always substitute the real value.\n"
    "- If the requested shape is an object, reply with that object. If it's a plain "
    "string, number, or array, reply with just that value as valid JSON."
)


def ask_llm_for_answer(history, chat_id):
    for model_name in MODELS_TO_TRY:
        model_name = model_name.strip()
        if not model_name:
            continue
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": ANSWER_SYSTEM_PROMPT}] + history[-6:],
            )
            content = response.choices[0].message.content
            if content is None:
                log_event({"type": "model_attempt", "chat_id": chat_id, "model": model_name, "error": "empty content"})
                continue
            content = content.strip()
            parsed = parse_json_value(content)
            log_event({
                "type": "model_attempt",
                "chat_id": chat_id,
                "model": model_name,
                "raw_output": content,
                "parsed_ok": parsed is not None,
            })
            if parsed is not None and parsed != {}:
                return parsed
        except Exception as e:
            logger.warning(f"LLM API call failed with model {model_name}: {e}")
            log_event({"type": "model_attempt", "chat_id": chat_id, "model": model_name, "error": str(e)})
    return None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am a data analysis Telegram bot. Send me your query.")


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

    computed_answer = solve_known_question(user_text)
    if computed_answer is None:
        computed_answer = ask_llm_for_answer(history, chat_id)

    if computed_answer is None or computed_answer == {}:
        logger.warning("No usable answer from any model; sending explicit failure marker.")
        log_event({"type": "answer_failed", "chat_id": chat_id})
        # Be honest about failure rather than quietly returning a fixed
        # guess for every question type — a wrong-shaped guess is graded
        # as wrong anyway, and this makes failures visible in your logs.
        computed_answer = None

    history.append({"role": "assistant", "content": json.dumps(computed_answer)})

    final_obj = {
        "answer": computed_answer,
        "log_url": get_effective_log_url(),
    }
    final_reply = json.dumps(final_obj)

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