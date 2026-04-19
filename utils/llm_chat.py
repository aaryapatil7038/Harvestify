import json
import os
from typing import Optional

import requests

from utils.farm_chat import DEFAULT_SUGGESTIONS


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _safe_json(value):
    try:
        return json.dumps(value or {}, ensure_ascii=False, indent=2)
    except Exception:
        return "{}"


def _build_system_prompt(lang):
    language_line = (
        "The current user question is in Marathi, so reply in Marathi only. Use clear, farmer-friendly Marathi."
        if lang == "mr"
        else "The current user question is in English, so reply in English. Be clear, practical, and farmer-friendly."
    )
    return (
        "You are Harvestify Farm Assistant, an in-app agricultural helper. "
        "Answer farming questions, explain how to use the Harvestify website, and use the provided app context when helpful. "
        "Prefer practical, direct answers over generic disclaimers. "
        "If the user asks about crop cultivation, fertilizer, irrigation, pests, disease, symptoms, soil, or season, answer with useful guidance. "
        "If exact diagnosis or dosage is uncertain, say what extra details are needed rather than refusing. "
        "When the user asks how to use the website, explain the steps clearly. "
        "Always match the language of the current user question, even if older chat history or app context uses another language. "
        f"{language_line} Keep answers concise but useful."
    )


def _build_user_prompt(message, lang, session_context, client_context, chat_history):
    return "\n\n".join([
        f"User language: {lang}",
        "Recent chat history:",
        _safe_json(chat_history[-8:] if isinstance(chat_history, list) else []),
        "Session context:",
        _safe_json(session_context),
        "Client context:",
        _safe_json(client_context),
        f"Current user question: {message}",
    ])


def generate_llm_farm_chat_reply(message, lang, session_context=None, client_context=None, chat_history=None):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    if not api_key or api_key.startswith("replace-with-") or not message:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.4,
        "max_tokens": 400,
        "messages": [
            {
                "role": "system",
                "content": _build_system_prompt(lang),
            },
            {
                "role": "user",
                "content": _build_user_prompt(
                    message=message,
                    lang=lang,
                    session_context=session_context or {},
                    client_context=client_context or {},
                    chat_history=chat_history or [],
                ),
            },
        ],
    }

    try:
        response = requests.post(
            OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        reply = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        reply_text = str(reply or "").strip()
        if not reply_text:
            return None
        return {
            "reply": reply_text,
            "suggestions": DEFAULT_SUGGESTIONS.get(lang, DEFAULT_SUGGESTIONS["en"]),
        }
    except requests.RequestException:
        return None
