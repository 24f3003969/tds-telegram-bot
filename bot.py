import json
import time
import os
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

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
        "You are a careful data analyst. The user's LAST message asks a data-analysis "
        "question and specifies the EXACT JSON schema/shape to reply with (e.g. {\"name\": \"...\"} or {\"answer\": {\"state\": \"...\"}}). "
        "Work out the correct answer using real data or arithmetic. "
        "CRITICAL REQUIREMENT: Your output MUST be a JSON object whose top-level and nested keys EXACTLY match the key names shown in the user's requested JSON template/example (e.g., if the example uses \"name\", you MUST use \"name\", NOT \"state\"). "
        "Do NOT add any extra keys, do NOT omit keys, and do NOT change key names. "
        "Reply with ONLY that exact JSON object and absolutely nothing else — no prose, no markdown code fences, just the raw JSON."
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system_prompt}] + history[-6:],
    )
    reply_text = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply_text})

    # Clean up markdown code blocks if the model wrapped its reply
    cleaned_text = reply_text
    if cleaned_text.startswith("```"):
        import re
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Model added extra text — try to pull out just the {...} part.
        start, end = reply_text.find("{"), reply_text.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(reply_text[start:end + 1])
        else:
            parsed = reply_text

    # Only set log_url if the question explicitly requested a log_url key in the reply
    if isinstance(parsed, dict) and "log_url" in parsed and LOG_URL != "PASTE_YOUR_PUBLIC_LOG_URL_HERE":
        parsed["log_url"] = LOG_URL

    final_reply = json.dumps(parsed) if isinstance(parsed, (dict, list)) else str(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)

app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
print("Bot is running... (Ctrl+C to stop)")
app.run_polling()
