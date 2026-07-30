import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters


# --- Minimal HTTP Server for Render Health Check ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def run_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()


# Start health check server in a background thread to pass Render Web Service checks
threading.Thread(target=run_health_check_server, daemon=True).start()
# --------------------------------------------------

# Load configuration from environment variables (do NOT hardcode secrets in source code)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
LOG_URL = os.environ.get("LOG_URL", "PASTE_YOUR_PUBLIC_LOG_URL_HERE")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set!")
if not AIPIPE_TOKEN:
    raise ValueError("AIPIPE_TOKEN environment variable is not set!")

client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)
LOG_FILE = "run.jsonl"

# Keeps the last few messages per chat, so multi-turn questions work —
# "answer the LAST message" still needs the earlier ones for context.
conversation_history = {}

def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    # Ask the AI to work out the answer. The system prompt tells it exactly how to
    # format the final reply — this is the part that MUST match what the question asked.
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
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "system", "content": system_prompt}] + history[-6:],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        reply_text = response.choices[0].message.content.strip()
    except Exception:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "system", "content": system_prompt}] + history[-6:],
            temperature=0.0
        )
        reply_text = response.choices[0].message.content.strip()

    history.append({"role": "assistant", "content": reply_text})

    # Clean up markdown code blocks if the model wrapped its reply
    cleaned_text = reply_text
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Model added extra text — try to pull out just the {...} part.
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

        # Always inject/ensure log_url if requested by question or if LOG_URL env var is configured
        is_log_url_requested = "log_url" in user_text.lower() or "log_url" in parsed
        if is_log_url_requested:
            effective_log_url = LOG_URL if (LOG_URL and LOG_URL != "PASTE_YOUR_PUBLIC_LOG_URL_HERE") else "https://raw.githubusercontent.com/username/repo/main/run.jsonl"
            parsed["log_url"] = effective_log_url

    final_reply = json.dumps(parsed) if isinstance(parsed, (dict, list)) else str(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)

app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
print("Bot is running... (Ctrl+C to stop)")
app.run_polling()
