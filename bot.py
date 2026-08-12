import json
import logging
import math
import os
import re
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal HTTP server: health check + serves run.jsonl publicly
# ---------------------------------------------------------------------------
LOG_FILE = "run.jsonl"
LOG_LOCK = threading.Lock()


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("/run.jsonl"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                with LOG_LOCK:
                    if os.path.exists(LOG_FILE):
                        with open(LOG_FILE, "rb") as f:
                            self.wfile.write(f.read())
                        return
            except Exception:
                logger.exception("Failed serving run.jsonl")
            self.wfile.write(b"")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def run_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()


threading.Thread(target=run_health_check_server, daemon=True).start()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("telegram_bot_token")
AIPIPE_TOKEN = (
    os.environ.get("AIPIPE_TOKEN")
    or os.environ.get("aipipe_token")
    or os.environ.get("OPENAI_API_KEY")
)
LOG_URL = os.environ.get("LOG_URL") or os.environ.get("log_url") or ""

if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN environment variable is not set!")
if not AIPIPE_TOKEN:
    logger.warning("AIPIPE_TOKEN environment variable is not set!")
if not LOG_URL:
    logger.warning("LOG_URL environment variable is not set!")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
    "AIPIPE_BASE_URL", "https://aipipe.org/openai/v1"
)
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=AIPIPE_TOKEN) if AIPIPE_TOKEN else None

MODELS_TO_TRY = [
    m.strip()
    for m in os.environ.get(
        "MODELS_TO_TRY", "gpt-4o-mini,gpt-4o,gpt-3.5-turbo,gemini-2.5-flash"
    ).split(",")
    if m.strip()
]

MAX_TOOL_ROUNDS = 6
conversation_history = {}

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log_event(event: dict):
    event["timestamp"] = time.time()
    try:
        with LOG_LOCK:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        logger.exception("Failed to write log event")


# ---------------------------------------------------------------------------
# Tools the model can call to actually do data analysis
# ---------------------------------------------------------------------------
def tool_fetch_url(url: str, max_chars: int = 20000) -> str:
    """Fetch a public URL (dataset, CSV, JSON, HTML page) and return text."""
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "data-analyst-bot/1.0"})
        resp.raise_for_status()
        text = resp.text
        truncated = len(text) > max_chars
        return json.dumps(
            {
                "status_code": resp.status_code,
                "content_type": resp.headers.get("Content-Type", ""),
                "truncated": truncated,
                "text": text[:max_chars],
            }
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def tool_run_python(code: str) -> str:
    """Execute short analysis code in a restricted scope. Code must set a
    variable named `result` (JSON-serializable) with the computed answer."""
    scope = {
        "math": math,
        "statistics": statistics,
        "json": json,
        "re": re,
        "requests": requests,
    }
    try:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code, scope)
        result = scope.get("result", None)
        try:
            json.dumps(result)
        except TypeError:
            result = str(result)
        return json.dumps({"stdout": buf.getvalue()[-4000:], "result": result})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a public URL (CSV/JSON/HTML dataset or webpage) and return its raw text content, so you can analyze real data instead of guessing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The public URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python to parse/compute over data you've fetched (e.g. CSV text). "
                "You have `math`, `statistics`, `json`, `re`, `requests` available. "
                "Your code MUST assign the final computed value to a variable named `result`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute."},
                },
                "required": ["code"],
            },
        },
    },
]

TOOL_IMPLS = {"fetch_url": tool_fetch_url, "run_python": tool_run_python}

SYSTEM_PROMPT = (
    "You are an expert data analyst AI assistant. The user's message contains a data-analysis "
    "question and specifies an EXACT JSON schema/shape to reply with.\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. If the question references a dataset or public source (e.g. MOSPI, data.gov.in, Census), "
    "use the fetch_url tool to actually download it, and run_python to parse/compute over it. "
    "Do NOT answer from memory/guesswork when a real source is available or implied.\n"
    "2. Analyze the question carefully; if it's multi-turn, the LAST message is the one to answer, "
    "using earlier turns as context.\n"
    "3. When you are ready, output ONLY a single JSON object matching the requested schema, and "
    "nothing else — no markdown fences, no prose before or after.\n"
    "4. Every value in the schema MUST be a real, computed answer — never leave a value as {}, "
    "null, an empty string, an empty list, or a literal placeholder like <num> or <state name>. "
    "If asked for a number, compute it. If asked for a category/name, name it.\n"
    "5. Do NOT include extra keys such as 'explanation', 'reasoning', 'notes', 'thought', or "
    "'comments' in the final JSON.\n"
    "6. If the request specifies 'answer' and 'log_url' keys, output both, both populated."
)


# ---------------------------------------------------------------------------
# Model call with tool-use loop + model/mode fallback
# ---------------------------------------------------------------------------
def call_model(history, chat_id):
    if not client:
        raise RuntimeError("OpenAI/AIPipe client is not configured (missing AIPIPE_TOKEN)")
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-20:]
            for _round in range(MAX_TOOL_ROUNDS):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=TOOLS_SPEC,
                )
                msg = response.choices[0].message

                if msg.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in msg.tool_calls
                            ],
                        }
                    )
                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        impl = TOOL_IMPLS.get(fn_name)
                        tool_result = impl(**args) if impl else json.dumps(
                            {"error": f"unknown tool {fn_name}"}
                        )
                        log_event(
                            {
                                "type": "tool_call",
                                "chat_id": chat_id,
                                "model": model_name,
                                "tool": fn_name,
                                "args": args,
                                "result_preview": str(tool_result)[:500],
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(tool_result)[:8000],
                            }
                        )
                    continue  # let the model see tool results and continue

                content = msg.content
                if content:
                    log_event(
                        {
                            "type": "model_attempt",
                            "chat_id": chat_id,
                            "model": model_name,
                            "raw_output": content,
                        }
                    )
                    return content.strip()
                break  # empty, no tool call -> try next model

        except Exception as e:
            last_error = e
            logger.warning(f"Model {model_name} failed: {e}")
            log_event(
                {"type": "model_attempt", "chat_id": chat_id, "model": model_name, "error": str(e)}
            )

    raise RuntimeError(f"All models failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Local fast-path for trivially inline-computable questions (best-effort only;
# NOT a hardcoded answer key — falls through to the LLM+tools for anything
# it doesn't recognize as a pure arithmetic transform over inline numbers)
# ---------------------------------------------------------------------------
def try_python_solver(user_text: str):
    text_lower = user_text.lower()
    match_mult = re.search(r"multiply\s+(?:each\s+input\s+)?by\s+([0-9.]+)", text_lower)
    match_array = re.search(r"\[([0-9\s,.\-]+)\]", user_text)
    if match_mult and match_array:
        try:
            factor = float(match_mult.group(1))
            nums = [
                float(x.strip()) if "." in x else int(x.strip())
                for x in match_array.group(1).split(",")
            ]
            decimals = 2
            match_round = re.search(r"round\s+to\s+([0-9]+)\s+decimal", text_lower)
            if match_round:
                decimals = int(match_round.group(1))
            values = [round(x * factor, decimals) for x in nums]
            return {"values": values}
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Robust JSON extraction from model output
# ---------------------------------------------------------------------------
def parse_llm_response(reply_text: str):
    if not reply_text:
        return None

    py_match = re.search(r"```python\s*(.*?)\s*```", reply_text, re.DOTALL)
    if py_match:
        scope = {"math": math, "statistics": statistics, "json": json, "re": re}
        try:
            exec(py_match.group(1), scope)
            if "result" in scope and isinstance(scope["result"], (dict, list)):
                return scope["result"]
        except Exception:
            pass

    cleaned = reply_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_b, end_b = reply_text.find("{"), reply_text.rfind("}")
    if start_b != -1 and end_b > start_b:
        try:
            return json.loads(reply_text[start_b : end_b + 1])
        except json.JSONDecodeError:
            pass

    start_sq, end_sq = reply_text.find("["), reply_text.rfind("]")
    if start_sq != -1 and end_sq > start_sq:
        try:
            return json.loads(reply_text[start_sq : end_sq + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Empty-answer detection
# ---------------------------------------------------------------------------
def is_answer_empty(parsed):
    if parsed is None:
        return True
    if isinstance(parsed, dict):
        if "answer" in parsed:
            return is_answer_empty(parsed["answer"])
        return len(parsed) == 0
    if isinstance(parsed, (list, str)):
        return len(parsed) == 0
    return False  # numbers/bools/populated containers are fine


# ---------------------------------------------------------------------------
# Assemble final {"answer": ..., "log_url": ...} shape
# ---------------------------------------------------------------------------
def format_final_reply(parsed, user_text: str, log_url: str):
    if parsed is None:
        parsed = {}

    is_log_url_requested = "log_url" in user_text.lower() or (
        isinstance(parsed, dict) and "log_url" in parsed
    )
    is_answer_wrapper_requested = (
        '"answer"' in user_text.lower()
        or "'answer'" in user_text.lower()
        or (isinstance(parsed, dict) and "answer" in parsed)
    )

    effective_log_url = log_url or LOG_URL

    if isinstance(parsed, dict):
        for unwanted in ["explanation", "reasoning", "notes", "thought", "comments", "confidence"]:
            parsed.pop(unwanted, None)

    if is_log_url_requested and is_answer_wrapper_requested:
        if isinstance(parsed, dict) and "answer" in parsed:
            answer_content = parsed["answer"]
        else:
            if isinstance(parsed, dict):
                parsed.pop("log_url", None)
            answer_content = parsed
        final_obj = {"answer": answer_content, "log_url": effective_log_url}
    elif is_log_url_requested:
        if isinstance(parsed, dict):
            if "answer" in parsed:
                final_obj = {"answer": parsed["answer"], "log_url": effective_log_url}
            else:
                final_obj = dict(parsed)
                final_obj["log_url"] = effective_log_url
        else:
            final_obj = {"answer": parsed, "log_url": effective_log_url}
    elif is_answer_wrapper_requested:
        final_obj = parsed if (isinstance(parsed, dict) and "answer" in parsed) else {"answer": parsed}
    else:
        if isinstance(parsed, dict) and "answer" in parsed and len(parsed) == 1:
            final_obj = parsed["answer"]
        else:
            final_obj = parsed

    return json.dumps(final_obj)


# ---------------------------------------------------------------------------
# Main processing: solve, validate non-empty, retry with nudge if empty
# ---------------------------------------------------------------------------
def process_message_logic(history, chat_id, user_text, log_url):
    solved = try_python_solver(user_text)
    raw_reply = ""

    if solved is not None:
        parsed = solved
    else:
        try:
            raw_reply = call_model(history, chat_id)
            parsed = parse_llm_response(raw_reply)

            if is_answer_empty(parsed):
                logger.warning("First pass returned empty answer for chat %s, retrying with nudge", chat_id)
                nudge_history = history + [
                    {"role": "assistant", "content": raw_reply or ""},
                    {
                        "role": "user",
                        "content": (
                            "Your last reply left the answer empty, null, or as a placeholder. "
                            "Use the fetch_url and run_python tools if you haven't already actually "
                            "retrieved and computed the answer. You must return a real, fully "
                            "computed value — never {}, null, or an unfilled placeholder. "
                            "Reply again with ONLY the completed JSON object."
                        ),
                    },
                ]
                raw_reply = call_model(nudge_history, chat_id)
                parsed = parse_llm_response(raw_reply)

        except Exception as e:
            logger.exception("call_model failed")
            log_event({"type": "answer_failed", "chat_id": chat_id, "error": str(e)})
            parsed = {}

    log_event(
        {
            "type": "parsed_result",
            "chat_id": chat_id,
            "parsed": parsed,
            "empty": is_answer_empty(parsed),
        }
    )
    return format_final_reply(parsed, user_text, log_url), raw_reply


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    final_reply, raw_reply = process_message_logic(history, chat_id, user_text, LOG_URL)
    history.append({"role": "assistant", "content": raw_reply or final_reply})

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(error_handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot is running... (Ctrl+C to stop)")
    app.run_polling(drop_pending_updates=True)